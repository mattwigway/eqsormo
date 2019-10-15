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

def clear_market (non_price_utilities, hhidx, choiceidx, supply, income, starting_price, price_income_transformation, price_income_params, budget_coef, max_rent_to_income=None, convergence_criterion=1e-6, maxiter=np.inf):
    '''
    Clear the market

    income and price should be by household index/choice index, respectively.
    '''
    price = starting_price

    alt_income = income[hhidx]

    if max_rent_to_income is not None:
        origNExcluded = np.sum(alt_income * max_rent_to_income <= price[choiceidx])
    with tqdm() as pbar:
        itr = 1
        while True:
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

            shares = np.bincount(choiceidx, weights=probs)

            prev_price = np.copy(price)

            maxdiff = np.max(np.abs(shares - supply))
            if maxdiff < convergence_criterion:
                #print('Prices converged after ' + str(itr) + ' iterations')
                return price

            if itr > maxiter:
                LOG.error(f'Prices FAILED TO CONVERGE in {maxiter} iterations!')
                return price

            # probably will need to remove if using numba
            maxpricediff = np.max(np.abs(price - prev_price))
            if max_rent_to_income is not None:
                deltaNExcluded = np.sum(alt_income * max_rent_to_income <= alt_price) - origNExcluded
            else:
                deltaNExcluded = 'n/a'
            pbar.set_postfix({'max_unit_diff': maxdiff, 'max_price_diff': maxpricediff, 'delta_n_excluded': deltaNExcluded}, refresh=False)

            #worst = np.argmax(np.abs(shares - supply))

            prev_price[:] = price

            # Forget where I saw this approach, but at each iteration average together all previous iterations
            # to prevent oscillation. In one of the papers - find citation later. Might be able to optimize this to make it converge faster
            # this implicitly assumes a perfect elasticity of price, but that assumption won't break it if it's wrong-just slow it down
            # 0.5 is not a magic value for this weight, it's likely that other values might work better.
            # also could estimate marginal effects of price and use those to make it solve in a smarter way
            # but this brute force approach does converge, and fast enough to be acceptable.
            wt = 0.75 # 1 / np.log(itr + 2)
            price = price * shares / supply * wt + price * (1 - wt)
            itr += 1
            pbar.update()


