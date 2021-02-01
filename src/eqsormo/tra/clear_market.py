#    Copyright 2019-2021 Matthew Wigginton Conway

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# Author: Matthew Wigginton Conway <matt@indicatrix.org>
#         School of Geographical Sciences and Urban Planning,
#         Arizona State University

import numpy as np

from logging import getLogger
import tempfile
import queue
import threading
import pandas as pd
import os
import uuid
import time
import scipy.optimize
from functools import lru_cache
from eqsormo.common.util import human_time, max_thread_count

LOG = getLogger(__name__)


class ClearMarket(object):
    def __init__(self, model, price_step=1e-5, maxiter=None):
        self.model = model
        self.price = model.price.loc[model.housing_xwalk.index].to_numpy()
        self.supply = model.weighted_supply.loc[model.housing_xwalk.index].to_numpy()
        self.fixed_price_index = model.housing_xwalk.loc[model.fixed_price]
        self.fixed_price = model.price.loc[model.fixed_price]
        self.price_step = price_step
        self.maxiter = maxiter
        self.alt_income = model.income.loc[model.hh_xwalk.index].to_numpy()[
            model.full_hhidx
        ]
        self.checkpoint_uuid = uuid.uuid4().hex

    def clear_market(self, diagonal_iterations=3):
        """
        Parameters
        ----------
        diagonal_iterations: int, optional
            Number of iterations to perform with the diagonal of the Jacobian before switching to the full Jacobian.
            The diagonal of the Jacobian is much faster to compute, but does not guarantee convergence, but generally moves in
            the right direction. Doing the first few interations with the diagonal can significantly speed convergence.
        """
        LOG.info("Clearing the market (everyone stand back)")
        LOG.info("")
        start_time = time.perf_counter()

        if self.model.endogenous_variable_defs is not None:
            raise ValueError("Endogeneous variables not supported in sorting")

        self.non_price_utilities = self.model.full_utility(include_budget=False)

        i = 0
        current_price = self.remove_fixed_price(self.price)

        LOG.info("Computing shares")
        shares = self.shares(current_price)
        alpha = 1
        while self.maxiter is None or i < self.maxiter:
            # make the logs more consistent
            i += 1
            LOG.info(f"market clearing: begin iteration {i}")
            excess_demand = shares - self.supply
            current_obj_val = np.sum(excess_demand ** 2)

            # since they always have to sum to 100% of hhs max will always be >= 0, and min <= 0
            LOG.info(
                f"Maximum overdemand: {np.max(excess_demand):.3f}, underdemand: {np.min(excess_demand):.3f}, SSE: {current_obj_val}"
            )
            LOG.info(
                f"Maximum overdemand: {np.max(excess_demand / self.supply) * 100:.3f}%, underdemand: {np.min(excess_demand / self.supply) * 100:.3f}%"
            )

            # update prices
            jacob = self.compute_derivatives(
                current_price, shares, diagonal_only=i <= diagonal_iterations
            )
            LOG.info("Inverting Jacobian")
            jacob_inv = np.linalg.inv(jacob)

            # this is 7.7 from Tra's dissertation
            price_delta_full = jacob_inv @ self.remove_fixed_price(excess_demand)

            # run a golden section-ish search to minimize excess demand as much as possible using this set of price deltas
            LOG.info(
                f"Previous alpha value: {alpha}. Using Brent's method to find optimal alpha given gradient."
            )

            # hacky, but use an LRU cache here to avoid recomputing shares after optimization converges
            @lru_cache(maxsize=10)
            def prices_and_shares_for_alpha(candidate_alpha):
                candidate_prices = current_price - price_delta_full * candidate_alpha
                candidate_shares = self.shares(candidate_prices)
                return candidate_prices, candidate_shares

            def obj_val(candidate_alpha):
                new_shares = prices_and_shares_for_alpha(candidate_alpha)[1]
                return np.sum((new_shares - self.supply) ** 2)

            # Do not run many iterations here, don't spend too much time doing this
            alpha_res = scipy.optimize.minimize_scalar(
                obj_val,
                bracket=(alpha, alpha * 0.9),
                tol=1e-3,
                options={"disp": True, "maxiter": 5, "xtol": 1e-3},
                method="brent",
            )
            alpha = (
                alpha_res.x
            )  # don't actually care if it converged, as long as it moved us closer

            new_price, new_shares = prices_and_shares_for_alpha(alpha)
            new_obj_val = np.sum((new_shares - self.supply) ** 2)

            LOG.info(
                f"Found optimal alpha {alpha}, moves SSE from {current_obj_val} to {new_obj_val}"
            )

            if not new_obj_val < current_obj_val:
                if i <= diagonal_iterations:
                    LOG.error(
                        f"With optimal alpha {alpha}, objective did not improve, using full Jacobian!"
                    )
                    diagonal_iterations = -1
                    continue
                else:
                    raise ValueError(f"Objective did not improve with alpha {alpha}!")

            current_price = new_price
            shares = new_shares

            self.save_price_checkpoint(current_price, i)

            if np.allclose(shares, self.supply):
                self.model.price = self.to_pandas_price(
                    self.add_fixed_price(current_price)
                )
                end_time = time.perf_counter()
                LOG.info(
                    f"Market clearing converged in {i} iterations after {human_time(end_time - start_time)}"
                )
                return True

        # can only get here if maxiter is reached
        else:
            end_time = time.perf_counter()
            LOG.info(
                f"Market clearing FAILED TO CONVERGE in {self.maxiter} iterations after {human_time(end_time - start_time)}."
            )
            return False

    def shares(self, price):
        budgets, feasible_alts = self.get_budgets(price)
        full_utilities = (
            self.non_price_utilities
            + self.model.first_stage_fit.params.budget * budgets
        )
        exp_utility = np.exp(full_utilities)
        del full_utilities, budgets
        # force choice probability to zero for infeasible alts
        exp_utility[~feasible_alts] = 0
        del feasible_alts

        if not np.all(np.isfinite(exp_utility)):
            raise FloatingPointError("Not all exp(utilities) are finite (scaling?)")

        expsums = np.bincount(self.model.full_hhidx, weights=exp_utility)
        probs = exp_utility / expsums[self.model.full_hhidx]

        if self.model.weights is not None:
            probs *= self.model.weights.loc[self.model.hh_xwalk.index].to_numpy()[
                self.model.full_hhidx
            ]

        shares = np.bincount(self.model.full_choiceidx, weights=probs)

        return shares

    def get_budgets(self, price):
        """
        Return the budgets as well as the feasible alternatives for a set of prices.
        """
        alt_price = self.add_fixed_price(price)[self.model.full_choiceidx]
        budget = np.zeros_like(self.alt_income)
        if self.model.max_rent_to_income is None:
            feasible_alts = np.full_like(alt_price, True)
        else:
            feasible_alts = self.alt_income * self.model.max_rent_to_income > alt_price

        budget = np.zeros_like(self.alt_income)
        budget[feasible_alts] = self.model.price_income_transformation.apply(
            self.alt_income[feasible_alts],
            alt_price[feasible_alts],  # TODO price income params
        )

        return budget, feasible_alts

    def remove_fixed_price(self, price):
        """
        From a full price vector, return a price vector with the fixed price removed, used in market clearing.
        Since one price is held constant, we don't feed it into the root-finding algorithm.
        """
        return price[
            np.r_[0 : self.fixed_price_index, self.fixed_price_index + 1 : len(price)]
        ]

    def add_fixed_price(self, price):
        """
        From a price vector resulting from root finding, add the fixed price back in
        """
        # insert the fixed price at location fixed_price_index
        return np.concatenate(
            (
                price[: self.fixed_price_index],
                [self.fixed_price],
                price[self.fixed_price_index :],
            )
        )

    def to_pandas_price(self, price):
        """
        Convert a price vector _which includes the fixed price_ back to Pandas format.
        """
        return pd.Series(price, index=self.model.housing_xwalk.index)

    def save_price_checkpoint(self, price, iteration):
        """
        Save a checkpoint of a price vector _which does not include a fixed prices_, so we can restart computation
        if the model crashes.
        """
        fname = (
            f"eqsormo-price-checkpoint-{self.checkpoint_uuid}-iter-{iteration}.parquet"
        )
        LOG.info(f"Saving price checkpoint to {fname}")
        self.to_pandas_price(self.add_fixed_price(price)).to_parquet(fname)

    def compute_derivatives(self, price, base_shares, diagonal_only=False):
        if diagonal_only:
            LOG.info("Ignoring off-diagonal elements of Jacobian")

        jacob = np.zeros((len(price), len(price)))

        budgets, feasible_alts = self.get_budgets(price)
        budget_coef = self.model.first_stage_fit.params.budget
        base_exp_utilities = np.exp(self.non_price_utilities + budget_coef * budgets)
        # exp utility of zero is out of choice set
        base_exp_utilities[~feasible_alts] = 0
        del budgets, feasible_alts

        if not np.all(np.isfinite(base_exp_utilities)):
            raise FloatingPointError("Not all exp(utilities) are finite (scaling?)")

        budget_step, feasible_alts_step = self.get_budgets(price + self.price_step)
        budget_coef = self.model.first_stage_fit.params.budget
        step_exp_utilities = np.exp(
            self.non_price_utilities + budget_coef * budget_step
        )
        # exp utility of zero is out of choice set
        step_exp_utilities[~feasible_alts_step] = 0
        del budget_step, feasible_alts_step

        if not np.all(np.isfinite(step_exp_utilities)):
            raise FloatingPointError(
                "Not all exp(step_utilities) are finite (scaling?)"
            )

        # memmap the base utilities so we can use copy-on-write
        fh, util_file = tempfile.mkstemp(prefix="eqsormo_mmap_", suffix=".bin")
        os.close(fh)
        try:
            mm = np.memmap(
                util_file, dtype="float64", mode="w+", shape=base_exp_utilities.shape
            )
            mm[:] = base_exp_utilities[:]
            mm.flush()

            # cache weights
            if self.model.weights is not None:
                alt_weights = self.model.weights.loc[
                    self.model.hh_xwalk.index
                ].to_numpy()[self.model.full_hhidx]

            task_queue = queue.Queue()
            result_queue = queue.Queue()
            stop_threads = threading.Event()

            def worker():
                while not stop_threads.is_set():
                    try:
                        jacidx, choice = task_queue.get(timeout=10)
                    except queue.Empty:
                        continue  # return to top of loop to check for stop_threads event
                    else:
                        exp_utilities = np.memmap(
                            util_file,
                            dtype="float64",
                            mode="c",
                            shape=step_exp_utilities.shape,
                        )
                        # save memory by just saving indices, not full bool array
                        # in the model in my dissertation, there are ~1000 choices, so this will be 1/1000 of the indices
                        # since an int is 32 bits (or maybe 64 since numpy arrays can be big), and a bool I believe takes up
                        # an entire byte, we come out ahead b/c 1/1000 * 32 < 8
                        choicemask = np.nonzero(self.model.full_choiceidx == choice)
                        exp_utilities[choicemask] = step_exp_utilities[choicemask]
                        util_sums = np.bincount(
                            self.model.full_hhidx, weights=exp_utilities
                        )

                        if diagonal_only:
                            probs = (
                                exp_utilities[choicemask]
                                / util_sums[self.model.full_hhidx[choicemask]]
                            )
                            if self.model.weights is not None:
                                probs *= alt_weights[choicemask]
                            share_step = np.sum(probs)
                            jaccol = np.zeros(len(price))
                            jaccol[jacidx] = (
                                share_step - base_shares[choice]
                            ) / self.price_step
                        else:
                            probs = exp_utilities / util_sums[self.model.full_hhidx]
                            if self.model.weights is not None:
                                probs *= alt_weights
                            share_step = np.bincount(
                                self.model.full_choiceidx, weights=probs
                            )
                            del probs
                            jaccol = self.remove_fixed_price(
                                (share_step - base_shares) / self.price_step
                            )

                        result_queue.put((jacidx, jaccol))
                        task_queue.task_done()

            def consumer():
                while not stop_threads.is_set():
                    try:
                        jacidx, jaccol = result_queue.get(timeout=10)
                    except queue.Empty:
                        continue
                    else:
                        if jacidx % 10 == 9:
                            LOG.info(f"computed derivative {jacidx + 1} / {len(price)}")
                        jacob[:, jacidx] = jaccol
                        result_queue.task_done()

            # might need something more complex here to account for the memory pressure of sorting. Even if you have
            # 16 cores, you might not have enough memory to compute 16 derivatives at once.
            nthreads = max_thread_count() if diagonal_only else 2
            LOG.info(f"computing derivatives using {nthreads} threads")

            # start threads
            for i in range(nthreads):
                threading.Thread(target=worker, daemon=False).start()

            # consumer thread
            threading.Thread(target=consumer, daemon=False).start()

            # fill queue
            # note that jacidx is the index in the Jacobian, which is not the same as choice which is the housing choice
            # b/c one housing choice is skipped in the Jacobian
            for jacidx, choice in enumerate(
                np.r_[
                    0 : self.fixed_price_index,
                    self.fixed_price_index + 1 : len(self.price),
                ]
            ):
                task_queue.put((jacidx, choice))

            # await completion
            task_queue.join()
            result_queue.join()

            assert not np.any(
                np.diag(jacob) >= 0
            ), "some diagonal elements of jacobian are nonnegative!"

            # signal threads to shut down
            stop_threads.set()

            return jacob
        finally:
            # clean up
            os.remove(util_file)
