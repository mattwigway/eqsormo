#    Copyright 2019 Matthew Wigginton Conway

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# Author: Matthew Wigginton Conway <matt@indicatrix.org>, School of Geographical Sciences and Urban Planning, Arizona State University

import numba
import numpy as np
from tqdm import tqdm

from logging import getLogger
LOG = getLogger(__name__)

def clear_market (non_price_utilities, hhidx, choiceidx, supply, income, starting_price, price_income_transformation,
        price_income_params, budget_coef, max_rent_to_income=None, convergence_criterion=1e-5, step=1e-2, maxiter=np.inf, weights=None):
    '''
    Clear the market

    income and price should be by household index/choice index, respectively.
    '''
    price = starting_price

    alt_income = income[hhidx]

    if max_rent_to_income is not None:
        origNExcluded = np.sum(alt_income * max_rent_to_income <= price[choiceidx])

    prev_price = np.copy(price)
    with tqdm() as pbar:
        itr = 0
        while True:
            itr += 1
            if itr > maxiter:
                LOG.error(f'Prices FAILED TO CONVERGE in {maxiter} iterations!')
                return price

            if max_rent_to_income is not None:
                # could cause oscillation if price keeps going above max income
                # Also, this will affect sorting equilibrium, because the price may effectively get fixed to the
                # 90th percentile income because it always moves off, and everything else equilibrate around it
                if np.any(price > np.max(income) * max_rent_to_income):
                    LOG.error('Some prices have exceeded max income - setting to be affordable to 90th percentile household to keep process going.')
                    price[price > np.max(income) * max_rent_to_income] = np.percentile(income, 0.9) * max_rent_to_income

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
                weights=weights
            )

            if np.any(shares == 0):
                raise ValueError('Some shares are zero.')

            excess_demand = shares - supply

            maxdiff = np.max(np.abs(shares - supply))
            #maxdiff = np.max(np.abs(excess_demand / supply))
            if maxdiff < convergence_criterion:
                LOG.info(f'Prices converged after {itr} iterations')
                return price

            # probably will need to remove if using numba
            maxpricediff = np.max(price - prev_price)
            minpricediff = np.min(price - prev_price)
            prev_price[:] = price
            if max_rent_to_income is not None:
                deltaNExcluded = np.sum(alt_income * max_rent_to_income <= price[choiceidx]) - origNExcluded
            else:
                deltaNExcluded = 'n/a'
            pbar.set_postfix({'max_unit_diff': maxdiff, 'max_price_diff': maxpricediff, 'min_price_diff': minpricediff, 'additional_excluded': deltaNExcluded}, refresh=False)

            # Use the approach defined in Tra (2007), page 108, eq. 7.7/7.7a, which is copied from Anas (1982)
            # first, compute derivative. Since the budget transformation is an arbitrary Python function, first compute its derivative.
            # since the budgets are independent between houses and between choosers, this is a fast numpy vectorized operation. We need a
            # loop to compute derivatives of budget. That can be done as a numpy vectorized operation
            alt_price = price[choiceidx]
            budget = price_income_transformation.apply(alt_income, alt_price, *price_income_params)
            price_step = 0.01
            budget_step = price_income_transformation.apply(alt_income, alt_price + price_step, *price_income_params)

            # fix one price from changing
            # TODO pick price in a smarter way (PUMA with least change?)
            # Appears to be causing convergence problems.
            #excess_demand[0] = 0

            deriv = compute_derivatives(price, alt_income, choiceidx, hhidx, non_price_utilities, budget, budget_step, price_step, budget_coef,
                shares, max_rent_to_income, weights)
        
            if not np.all(deriv < 0):
                raise ValueError('some derivatives of price are nonnegative')

            # this is 7.7a from the paper
            price = price - excess_demand / deriv

            pbar.update()

@numba.jit(nopython=True)
def compute_derivatives (price, alt_income, choiceidx, hhidx, non_price_utilities, budget, budget_step, price_step, budget_coef, base_shares, max_rent_to_income, weights):
    deriv = np.zeros_like(price)

    sim_budget = np.copy(budget)
    sim_price = np.copy(price[choiceidx])

    for i in range(len(price)):
        sim_budget[:] = budget
        sim_budget[choiceidx == i] = budget_step[choiceidx == i]

        # only used in places where price exceeds budget
        sim_price[:] = sim_price
        sim_price[choiceidx == i] += price_step

        full_utilities = non_price_utilities + budget_coef * sim_budget
        exp_utility = np.exp(full_utilities)

        if max_rent_to_income is not None:
            # setting expUtility to zero means choice probability will be 0, and choice will also not be added to the logsum
            exp_utility[alt_income * max_rent_to_income <= sim_price] = 0

        if not np.all(np.isfinite(exp_utility)):
            print('Not all exp(utilities) are finite (scaling?)')
            return np.zeros(0) # will cause error, and hopefully someone will see message above - work around numba limitation on raising

        logsums = np.bincount(hhidx, weights=exp_utility)
        probs = exp_utility / logsums[hhidx]

        if weights is not None:
            probs *= weights[hhidx]

        share_step = np.sum(probs[choiceidx == i])
        deriv[i] = (share_step - base_shares[i]) / price_step
    
    return deriv
    
def compute_shares (price, supply, alt_income, choiceidx, hhidx, non_price_utilities, price_income_transformation, price_income_params, budget_coef,
        max_rent_to_income, weights):
    alt_price = price[choiceidx]
    budgets = price_income_transformation.apply(alt_income, alt_price, *price_income_params)
    full_utilities = non_price_utilities + budget_coef * budgets
    expUtility = np.exp(full_utilities)

    if max_rent_to_income is not None:
        # setting expUtility to zero means choice probability will be 0, and choice will also not be added to the logsum
        expUtility[alt_income * max_rent_to_income <= alt_price] = 0

    if not np.all(np.isfinite(expUtility)):
        print('Not all exp(utilities) are finite (scaling?)')
        return np.array([]) # will cause error, and hopefully someone will see message above - work around numba limitation on raising

    logsums = np.bincount(hhidx, weights=expUtility)
    probs = expUtility / logsums[hhidx]

    if weights is not None:
        probs *= weights[hhidx]

    return np.bincount(choiceidx, weights=probs)
