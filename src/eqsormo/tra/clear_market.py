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
from eqsormo.common import nesterov
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

        LOG.info("Computing shares using gradient descent")
        shares = self.shares(current_price)

        # used in Nesterov's acceleration, see below
        prev_price_gd = current_price
        alpha = 1
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

            search_dir = self.remove_fixed_price(excess_demand)

            current_obj_val = np.sum(excess_demand ** 2)
            while True:
                LOG.info("computing new prices and market shares")

                # Use Nesterov's acceleration for optimal gradient descent
                # see https://blogs.princeton.edu/imabandit/2013/04/01/acceleratedgradientdescent/
                # start with base gradient descent - this is y_s in the blog post
                new_price_gd = current_price - search_dir * alpha
                # and slide a bit further - note that this is off-by-one because we use 1-based indices
                # for the logs
                new_price = (1 - nesterov.gamma(i)) * new_price_gd + nesterov.gamma(
                    i - 1
                ) * current_price

                new_shares = self.shares(new_price)
                new_obj_val = np.sum((new_shares - self.supply) ** 2)
                if new_obj_val < current_obj_val:
                    shares = new_shares
                    current_price = new_price
                    prev_price_gd = new_price_gd
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
