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
import time
import multiprocessing
from eqsormo.common.util import human_time

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

    def clear_market(self):
        LOG.info("Clearing the market (everyone stand back)")
        start_time = time.perf_counter()

        if self.model.endogenous_variable_defs is not None:
            raise ValueError("Endogeneous variables not supported in sorting")

        self.non_price_utilities = self.model.full_utility(include_budget=False)

        i = 0
        current_price = self.remove_fixed_price(self.price)

        LOG.info("Computing shares")
        shares = self.shares(current_price)

        # start with alpha 0.5 since we're ignoring the off-diagonal elements of the jacobian and thus
        # using just the estimated derivative will always overshoot because of the IIA property - increasing
        # the price for one property will increase the demand for all other properties. So the diagonal is only
        # half the story of market clearing.
        alpha = 0.5
        while self.maxiter is None or i < self.maxiter:
            # make the logs more consistent
            i += 1
            LOG.info(f"market clearing: begin iteration {i}")
            excess_demand = shares - self.supply
            # since they always have to sum to 100% of hhs max will always be >= 0, and min <= 0
            LOG.info(
                f"Maximum overdemand: {np.max(excess_demand):.3f}, underdemand: {np.min(excess_demand):.3f}"
            )
            LOG.info(
                f"Maximum overdemand: {np.max(excess_demand / self.supply) * 100:.3f}%, underdemand: {np.min(excess_demand / self.supply) * 100:.3f}%"
            )

            # update prices
            jacob_diag = self.compute_derivatives(current_price, shares)

            LOG.info("Computing full Jacobian from diagonal")
            jacob = self.full_jacobian_from_diagonal(jacob_diag, shares)
            LOG.info("Inverting Jacobian")
            jacob_inv = np.linalg.inv(jacob)

            # this is 7.7 from Tra's dissertation
            price_delta_full = price_delta_full = jacob_inv @ self.remove_fixed_price(excess_demand)
            current_obj_val = np.sum(excess_demand ** 2)
            while True:
                LOG.info("computing new prices and market shares")
                new_price = current_price - price_delta_full * alpha
                new_shares = self.shares(new_price)
                new_obj_val = np.sum((new_shares - self.supply) ** 2)
                if new_obj_val < current_obj_val:
                    shares = new_shares
                    current_price = new_price
                    break
                else:
                    # this is kind of a backtracking line search - if moving by alpha did not move us closer to
                    # convergence, don't move as far. Thanks to Sam Zhang for the tip here.
                    LOG.info(
                        f"moving along gradient by alpha {alpha} did not improve objective, setting alpha to {alpha / 2}"
                    )
                    alpha /= 2
                    continue

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

    def compute_derivatives(self, price, base_shares):
        jacob_diag = np.zeros_like(price)

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
                        # since an int is 32 bytes (or maybe 64 since numpy arrays can be big), and a bool I believe takes up
                        # an entire byte, we come out ahead b/c 1/1000 * 32 < 8
                        choicemask = np.nonzero(self.model.full_choiceidx == choice)
                        exp_utilities[choicemask] = step_exp_utilities[choicemask]
                        util_sums = np.bincount(
                            self.model.full_hhidx, weights=exp_utilities
                        )
                        probs = (
                            exp_utilities[choicemask]
                            / util_sums[self.model.full_hhidx[choicemask]]
                        )
                        if self.model.weights is not None:
                            probs *= alt_weights[choicemask]
                        share_step = np.sum(probs)
                        del probs
                        jacelem = (share_step - base_shares[choice]) / self.price_step

                        result_queue.put((jacidx, jacelem))
                        task_queue.task_done()

            def consumer():
                while not stop_threads.is_set():
                    try:
                        jacidx, jacelem = result_queue.get(timeout=10)
                    except queue.Empty:
                        continue
                    else:
                        if jacidx % 10 == 9:
                            LOG.info(f"computed derivative {jacidx + 1} / {len(price)}")
                        jacob_diag[jacidx] = jacelem
                        result_queue.task_done()

            # might need something more complex here to account for the memory pressure of sorting. Even if you have
            # 16 cores, you might not have enough memory to compute 16 derivatives at once.
            nthreads = multiprocessing.cpu_count()
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
                jacob_diag >= 0
            ), "some diagonal elements of jacobian are nonnegative!"

            # signal threads to shut down
            stop_threads.set()

            return jacob_diag
        finally:
            # clean up
            os.remove(util_file)

    def full_jacobian_from_diagonal (self, jac_diag, mkt_shares):
        """
        Because of the independence of irrelevant alternatives property of the multinomial logit model used to
        forecast demand, it is possible to derive the full Jacobian matrix for the demand system from the diagonal
        of the Jacobian. Numerically approximating the diagonal of the Jacobian takes about ten minutes for a full
        Southern California model, whereas approximating the full Jacobian takes about two hours. Computing the
        full Jacobian from the diagonal takes a negligible amount of time (milliseconds). However, using the full
        Jacobian makes the model converge _much_ faster than using just the diagonal, which Tra (2007) proposed to
        ease computation. However, because of this nifty property of the MNL model, we can have our cake and eat it
        too---get the per-iteration performance of the diagonal Jacobian, with the attractive convergence properties
        of using the full Jacobian.

        So here's how it works: in a discrete choice model, when demand for a particular alternative changes by x, the sum of demand for all other
        alternatives must change by -x (to keep all choice probabilities summing to 1 within each decisionmaking unit).
        However, the independence of irrelevant alternatives properties of the MNL makes a stronger assertion: when demand for
        a particular alternative changes by, the demand for all other alternatives change _in proportion to their market share_.
        Thus, the off-diagonal elements of the Jacobian are simply J[j,i] = -J[i,i] * S[j] / (1 - S[i]), where J is the
        Jacobian and S is the market share.
        """

        jacob = np.zeros((len(jac_diag), len(jac_diag)))
        
        mkt_shares_no_fixed = self.remove_fixed_price(mkt_shares)

        for i in range(len(jac_diag)):
            # There might be a way to do this faster, without a loop
            jacob[i,:] = -jac_diag[i] * mkt_shares_no_fixed / (1 - mkt_shares_no_fixed[i])

        np.fill_diagonal(jacob, jacob_diag)
        return jacob
