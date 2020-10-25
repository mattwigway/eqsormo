#    Copyright 2020 Matthew Wigginton Conway

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

from eqsormo.tra import price_income
import numpy as np
import scipy.stats
import pandas as pd

income = np.array([100, 150, 30, 20, 70])
price = np.array([8, 12, 14, 9, 10])


def test_logdiff():
    assert np.allclose(
        price_income.logdiff.apply(income, price), np.log(income - price)
    )


def test_sqrtdiff():
    assert np.allclose(
        price_income.sqrtdiff.apply(income, price), np.sqrt(income - price)
    )


def test_normdiff():
    assert np.allclose(
        price_income.normdiff.apply(income, price, 200),
        scipy.stats.norm.cdf(income - price, scale=200),
    )
    assert np.allclose(
        price_income.normdiff.apply(income, price, 300),
        scipy.stats.norm.cdf(income - price, scale=300),
    )
    # make sure the scale param has an effect
    assert not np.allclose(
        price_income.normdiff.apply(income, price, 300),
        price_income.normdiff.apply(income, price, 200),
    )
    # make sure it works with price > income - this transformation can handle this, unlike the other transformations
    bigprice = np.array([8, 12, 14, 30, 10])
    assert np.any(bigprice > income)
    assert not pd.isnull(price_income.normdiff.apply(income, bigprice, 200)).any()


def test_boxcox():
    assert np.allclose(
        price_income.boxcoxdiff.apply(income, price, 1),
        scipy.stats.boxcox(income - price, lmbda=1),
    )
    assert np.allclose(
        price_income.boxcoxdiff.apply(income, price, 3),
        scipy.stats.boxcox(income - price, lmbda=3),
    )
    assert not np.allclose(
        price_income.boxcoxdiff.apply(income, price, 1),
        price_income.boxcoxdiff.apply(income, price, 3),
    )
    # TODO test behavior when price >= income
