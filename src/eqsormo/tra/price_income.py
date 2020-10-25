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

"""
Functional forms for price and income
"""

import numpy as np
import scipy.stats


class FunctionalForm(object):
    def __init__(self, func, name, n_params=0, starting_values=None, param_names=None):
        """
        Create a new functional form. Pass in a function of (self, income, price, *args) that returns a transformed
        income/price variable. *args is the parameters for the function, and will have length n_params, which will start at
        the values in starting_values, which are then estimated in the maximum likelihood estimation. Note that income and price
        are numpy arrays or Pandas series, not scalars.
        """
        self.apply = func
        self.n_params = n_params
        self.name = name

        if param_names is None:
            param_names = [f"price_income_param_{i}" for i in range(n_params)]
        self.param_names = param_names

        if starting_values is None:
            starting_values = np.zeros(n_params)
        self.starting_values = starting_values

        assert len(param_names) == n_params
        assert len(starting_values) == n_params

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


# Log difference, i.e. ln(income - price), as used in Tra (2010)
# TODO: price > income? or drop from choice set
logdiff = FunctionalForm(
    lambda income, price: np.log(income - price), "log(income - price)"
)

# normcdf, kinda like log but defined below zero
# assumes ppl are most price sensitive where budget = price
normdiff = FunctionalForm(
    lambda income, price, scale: scipy.stats.norm.cdf(income - price, scale=scale),
    "normcdf(income - price)",
    n_params=1,
    starting_values=np.array([1]),
    param_names=["normcdf_sd"],
)

# square root of difference, as used in Tra (2010) robustness checks
sqrtdiff = FunctionalForm(
    lambda income, price: np.sqrt(income - price), "sqrt(income - price)"
)

# box-cox transform of income - price
# Box and Cox (1964)
# NB any price > income should be taken care of by a max_rent_to_income parameter
# TODO can we remove the max rent to income param here?
boxcoxdiff = FunctionalForm(
    lambda income, price, lmbda: scipy.stats.boxcox(
        np.maximum(income - price, 1), lmbda=lmbda
    ),
    "boxcox(income - price)",
    n_params=1,
    param_names=["boxcox_lambda"],
)

# Form suggested by Stephane Hess in his advanced discrete choice modelling class, London, UK, July 2019
# Note that this will not behave right if called on a subset of the data, since it uses the mean of income
# Removed because these functions are now always called on subsets
# hess = FunctionalForm(lambda income, price, hess_lambda: ((income / np.mean(income)) ** hess_lambda) * price,
#     'Hess: (income / mean(income)) ^ lambda * price', n_params=1, param_names=['hess_lambda'], starting_values=[-1])
