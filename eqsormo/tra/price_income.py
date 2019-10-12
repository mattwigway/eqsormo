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

'''
Functional forms for price and income
'''

import numpy as np
import scipy.stats

class FunctionalForm(object):
    def __init__ (self, func, n_params=0, starting_values=None, param_names=None):
        '''
        Create a new functional form. Pass in a function of (self, income, price, *args) that returns a transformed
        income/price variable. *args is the parameters for the function, and will have length n_params, which will start at
        the values in starting_values, which are then estimated in the maximum likelihood estimation. Note that income and price
        are numpy arrays or Pandas series, not scalars.
        '''
        self.apply = func
        self.n_params = n_params

        if param_names is None:
            param_names = [f'price_income_param_{i}' for i in range(n_params)]
        self.param_names = param_names
        
        if starting_values is None:
            starting_values = np.zeros(n_params)        
        self.starting_values = starting_values

        assert len(param_names) == n_params
        assert len(starting_values) == n_params


# Log difference, i.e. ln(income - price), as used in Tra (2010)
# TODO: price > income? or drop from choice set
logdiff = FunctionalForm(lambda income, price: np.log(income - price))

# square root of difference, as used in Tra (2010) robustness checks
sqrtdiff = FunctionalForm(lambda income, price: np.sqrt(income - price))

# box-cox transform of income - price
# Box and Cox (1964)
boxcoxdiff = FunctionalForm(lambda income, price, lmbda: scipy.stats.boxcox(income - price, lmbda=lmbda), n_params=1, param_names=['boxcox_lambda'])