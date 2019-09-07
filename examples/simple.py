# a simple example application of EqSorMo

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

import pandas as pd
import numpy as np
from sys import path, argv
import os.path
path.append(os.path.dirname(os.path.dirname(os.path.abspath(argv[0]))))
import eqsormo
eqsormo.enable_logging()

alt = pd.read_csv('data/alternatives.csv').set_index('choice')
hh = pd.read_csv('data/hh.csv')

hh['hhincome'] /= 1000
alt['nbhd_median_income'] /= 1000

mod = eqsormo.SortingModel(
    altHousing=alt[[]], # no structure specific attributes
    altNeighborhood=alt[['nbhd_median_income', 'nbhd_mean_hhsize']],
    altHedonic=alt[['nbhd_median_income', 'nbhd_mean_hhsize', 'singleFamily']],
    altPrice=alt.rentgrs,
    hh = hh[['hhincome', 'college', 'numprec']],
    hhChoice=hh.choice,
    interactions=[
        ('hhincome', 'nbhd_median_income'),
        ('numprec', 'nbhd_mean_hhsize'),
        ('college', 'nbhd_median_income')
    ],
    initialPriceCoef=-0.1,
    sampleAlternatives=10
)

np.random.seed(2832)
mod.fit()

print(mod.summary())