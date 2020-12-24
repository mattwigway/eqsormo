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
        origNExcluded = np.sum(alt_income * max_rent_to_income <= price[choiceidx])

    prev_price = np.copy(price)

    if max_rent_to_income is not None:
        # could cause oscillation if price keeps going above max income
        # Also, this will affect sorting equilibrium, because the price may effectively get fixed to the
        # 90th percentile income because it always moves off, and everything else equilibrate around it
        if np.any(price > np.max(income) * max_rent_to_income):
            LOG.error(
                "Some prices have exceeded max income - setting to be affordable to 90th"
                + "percentile household to keep process going."
            )
            price[price > np.max(income) * max_rent_to_income] = (
                np.percentile(income, 0.9) * max_rent_to_income
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
        return price, False

    excess_demand = shares - supply

    # maxdiff = np.max(np.abs(shares - supply) / supply)
    maxdiff = np.max(np.abs(excess_demand / supply))

    # probably will need to remove if using numba
    maxpricediff = np.max(price - prev_price)
    minpricediff = np.min(price - prev_price)
    prev_price[:] = price
    if max_rent_to_income is not None:
        deltaNExcluded = (
            np.sum(alt_income * max_rent_to_income <= price[choiceidx]) - origNExcluded
        )
    else:
        deltaNExcluded = "n/a"

    # Use the approach defined in Tra (2007), page 108, eq. 7.7/7.7a, which is copied from Anas (1982)
    # first, compute derivative. Since the budget transformation is an arbitrary Python function,
    # first compute its derivative.
    # since the budgets are independent between houses and between choosers, this is a fast numpy
    # vectorized operation. We need a loop to compute derivatives of budget. That can be done as a
    # numpy vectorized operation. TODO this doesn't make sense - maybe using vectorize to mean diff
    # things, i.e. actually vectorized vs np.vectorize?
    alt_price = price[choiceidx]

    budget = np.zeros_like(alt_income)
    if max_rent_to_income is None:
        feasible_alts = np.full_like(alt_price, True)
    else:
        feasible_alts = alt_income * max_rent_to_income > alt_price

    budget = np.full_like(alt_income, np.nan)
    budget[feasible_alts] = price_income_transformation.apply(
        alt_income[feasible_alts], alt_price[feasible_alts], *price_income_params
    )
    budget_step = np.full_like(alt_income, np.nan)

    if max_rent_to_income is None:
        feasible_alts_step = np.full_like(alt_price, True)
    else:
        feasible_alts_step = alt_income * max_rent_to_income > alt_price
    budget_step[feasible_alts] = price_income_transformation.apply(
        alt_income[feasible_alts_step],
        alt_price[feasible_alts_step] + step,
        *price_income_params,
    )

    # fix one price from changing
    # TODO pick price in a smarter way (PUMA with least change?)
    # Appears to be causing convergence problems.
    # excess_demand[0] = 0

    # NB derivatives are ill-behaved at the boundary where a choice enters a household's choice set due to
    # price dropping below their income. We assume that what is in each household's choice set is
    # nonchanging for the purpose of calculating derivatives.
    # Nope, that won't work.
    # TODO is the above comment correct? how is this handled now?
    deriv = compute_derivatives(
        price,
        alt_income,
        choiceidx,
        hhidx,
        non_price_utilities,
        budget,
        budget_step,
        step,
        budget_coef,
        shares,
        max_rent_to_income,
        feasible_alts,
        feasible_alts_step,
        weights,
    )

    if not np.all(deriv < 0):
        raise ValueError("some derivatives of price are nonnegative")

    # this is 7.7a from Tra's dissertation
    price = price - (excess_demand / deriv)
    LOG.info(
        f"Max unit diff: {maxdiff}, max price diff: {maxpricediff}, "
        + f"min price diff: {minpricediff}, additional excluded due to rent/income ration {deltaNExcluded}"
    )

    return price, False  # not converged yet (or we don't know anyhow)


# @numba.jit(nopython=True) seems to slow things down but I may just not be letting it warm up
def compute_derivatives(
    price,
    alt_income,
    choiceidx,
    hhidx,
    non_price_utilities,
    budget,
    budget_step,
    price_step,
    budget_coef,
    base_shares,
    max_rent_to_income,
    feasible_alts,
    feasible_alts_step,
    weights,
):
    deriv = np.zeros_like(price)

    base_exp_utilities = np.exp(non_price_utilities + budget_coef * budget)
    # exp utility of zero is out of choice set
    base_exp_utilities[~feasible_alts] = 0
    exp_utilities = np.copy(base_exp_utilities)
    step_exp_utilities = np.exp(non_price_utilities + budget_coef * budget_step)
    step_exp_utilities[~feasible_alts_step] = 0

    if not np.all(np.isfinite(exp_utilities)) or not np.all(
        np.isfinite(step_exp_utilities)
    ):
        raise FloatingPointError("Not all exp(utilities) are finite (scaling?)")

    # cache weights
    if weights is not None:
        alt_weights = weights[hhidx]

    for i in range(len(price)):
        choicemask = choiceidx == i
        exp_utilities[choicemask] = step_exp_utilities[choicemask]
        util_sums = np.bincount(hhidx, weights=exp_utilities)
        probs = exp_utilities[choicemask] / util_sums[hhidx[choicemask]]
        if weights is not None:
            probs *= alt_weights[choicemask]
        share_step = np.sum(probs)
        deriv[i] = (share_step - base_shares[i]) / price_step
        assert deriv[i] < 0, f"derivative of price is nonnegative!"
        exp_utilities[choicemask] = base_exp_utilities[choicemask]

    return deriv


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
