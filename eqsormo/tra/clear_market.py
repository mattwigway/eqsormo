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

def clear_market (non_price_utilities, hhidx, choiceidx, supply, income, starting_price, price_income_transformation, price_income_params, budget_coef, convergence_criterion=1e-6):
    # don't pass Python objects (classes) around
    return _clear_market(non_price_utilities, hhidx, choiceidx, supply, income, starting_price,
        price_income_transformation.apply, price_income_params, budget_coef, convergence_criterion)

#@numba.jit(nopython=True)
def _clear_market (non_price_utilities, hhidx, choiceidx, supply, income, starting_price, price_income_function, price_income_params, budget_coef, convergence_criterion):
    '''
    Clear the market

    income and price should be by household index/choice index, respectively.
    '''
    price = starting_price

    alt_income = income[hhidx]

    with tqdm() as pbar:
        itr = 1
        while True:
            alt_price = price[choiceidx]
            budgets = price_income_function(alt_income, alt_price, *price_income_params)
            full_utilities = non_price_utilities + budget_coef * budgets
            expUtility = np.exp(full_utilities)

            if not np.all(np.isfinite(expUtility)):
                print('Not all utilities are finite (scaling?)')
                return np.array([]) # will cause error, and hopefully someone will see message above - work around numba limitation on raising

            logsums = np.bincount(hhidx, weights=expUtility)
            probs = expUtility / logsums[hhidx]

            shares = np.bincount(choiceidx, weights=probs)

            maxdiff = np.max(np.abs(shares - supply))
            if maxdiff < convergence_criterion:
                #print('Prices converged after ' + str(itr) + ' iterations')
                return price

            # probably will need to remove if using numba
            pbar.set_postfix({'maxdiff': maxdiff}, refresh=False)

            #worst = np.argmax(np.abs(shares - supply))

            # Forget where I saw this approach, but at each iteration average together all previous iterations
            # to prevent oscillation. In one of the papers - find citation later. Might be able to optimize this to make it converge faster
            # this implicitly assumes a perfect elasticity of price, but that assumption won't break it if it's wrong-just slow it down
            wt = 1 / np.log(itr + 2)
            price = price * shares / supply * wt + price * (1 - wt)
            itr += 1
            pbar.update()


