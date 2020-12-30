#    Copyright 2019-2020 Matthew Wigginton Conway

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
import multiprocessing
import os
import pandas as pd

LOG = getLogger(__name__)


def clear_market_iter(
    non_price_utilities,
    hhidx,
    choiceidx,
    supply,
    income,
    starting_price,
    price_income_transformation,
    price_income_params,
    budget_coef,
    max_rent_to_income=None,
    convergence_criterion=1e-4,
    step=1e-2,
    weights=None,
    fixed_price=0,
    speed_control=0.25,
):
    """
    Run one iteration of the market-clearing algorithm.

    income and price should be by household index/choice index, respectively.

    returns price after next iteration, and flag for whether prices have converged. Note that once
    prices converge, one more call is necessary to confirm convergence, to avoid additional computation
    of market shares to check convergence after prices are updated.
    """
    price = starting_price

    alt_income = income[hhidx]

    if max_rent_to_income is not None:
        # could cause oscillation if price keeps going above max income
        # Also, this will affect sorting equilibrium, because the price may effectively get fixed to the
        # 90th percentile income because it always moves off, and everything else equilibrate around it
        if np.any(price > np.max(income) * max_rent_to_income):
            raise ValueError(
                "Some prices have exceeded max income - setting to be affordable to 90th"
                + "percentile household to keep process going."
            )

    shares = compute_shares(
        price=price,
        supply=supply,
        alt_income=alt_income,
        choiceidx=choiceidx,
        hhidx=hhidx,
        non_price_utilities=non_price_utilities,
        price_income_transformation=price_income_transformation,
        price_income_params=price_income_params,
        budget_coef=budget_coef,
        max_rent_to_income=max_rent_to_income,
        weights=weights,
    )

    if np.any(shares == 0):
        raise ValueError("Some shares are zero.")

    if np.allclose(shares, supply):
        return price, True

    assert np.allclose(
        np.sum(shares), np.sum(supply)
    ), "shares and supply totals do not match"

    LOG.info(f"Ratios of market share to supply: {(shares / supply).describe()}")

    # same logic as the ASC-finding algorithm, move prices that are too high lower and vice-versa
    # it's possible however that since prices are inside a log and multiplied by a coef that there
    # are some situations this could not climb out of.
    # I think that when using this formulation to compute ASCs, ln (share / supply) is equivalent to finding the ASC
    # that would produce the correct market share holding all other ASCs constant, and continuing on to convergence---
    # not that different than the approach using the derivative proposed in Tra's dissertation, eq. 7.7/7.7a, although
    # without using a straight line approximation. Here, there's no theory other than that when the ratio is > 1, reduce
    # price, and do the opposite when it's less. But since (I believe) there is a single unique price vector that clears
    # the market as long as we hold one price fixed, how you get to equilibrium doesn't matter, as long as you get there.
    orig_fixed_price = price[fixed_price]
    price -= np.log(shares / supply)
    price[fixed_price] = orig_fixed_price
    return price, False


def compute_shares(
    price,
    supply,
    alt_income,
    choiceidx,
    hhidx,
    non_price_utilities,
    price_income_transformation,
    price_income_params,
    budget_coef,
    max_rent_to_income,
    weights,
):
    alt_price = price[choiceidx]

    # unfortunately feasible alts does need to be recalculated on each iter as prices may change
    if max_rent_to_income is None:
        feasible_alts = np.full_like(alt_price, True)
    else:
        feasible_alts = alt_income * max_rent_to_income > alt_price

    budgets = np.zeros_like(alt_price)
    budgets[feasible_alts] = price_income_transformation.apply(
        alt_income[feasible_alts], alt_price[feasible_alts], *price_income_params
    )
    full_utilities = non_price_utilities + budget_coef * budgets
    exp_utility = np.exp(full_utilities)
    del full_utilities, budgets
    exp_utility[
        ~feasible_alts
    ] = 0  # will force choice probability to zero for infeasible alts
    del feasible_alts

    if not np.all(np.isfinite(exp_utility)):
        raise FloatingPointError("Not all exp(utilities) are finite (scaling?)")

    expsums = np.bincount(hhidx, weights=exp_utility)
    probs = exp_utility / expsums[hhidx]

    if weights is not None:
        probs *= weights[hhidx]

    return np.bincount(choiceidx, weights=probs)
