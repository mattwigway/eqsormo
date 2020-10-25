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

# TODO is this file still needed?

import numpy as np
from tqdm import tqdm
import pandas as pd


def compute_ascs(data, params, supply, starting_values, convergence_criterion=1e-6):
    """
    Compute the ASCs implied by params for a multinomial logit model using data so that supply is correct.
    Data should be indexed by decisionmakers, then by choices.

    The algorithm herein is defined in Bayer et al. (2004), in equation 16. Specifically, for any set of parameters, the set
    starting values for the ASCs can be moved closer to the true values using the formula

    ASC_h(t+1) = ASC_h(t) - ln(sum(P_h) - S_h) where P_h is the probability of a particular decisionmaker (subscript suppressed) choosing outcome h.

    Repeat this process until the supply and demand are within convergence_criterion for every choice
    """

    # Utilities without the ASCs
    if isinstance(params, pd.Series):
        baseUtilities = (
            data[params.index].multiply(params, axis="columns").sum(axis="columns")
        )
    else:  # assume numpy array
        baseUtilities = data.multiply(params, axis="columns").sum(axis="columns")

    ascs = starting_values

    while True:
        firstStageUtilities = (
            baseUtilities
            + ascs.loc[baseUtilities.index.get_level_values("choice")].values
        )
        expUtils = np.exp(firstStageUtilities)
        firstStageShares = (
            (expUtils / expUtils.groupby(level=0).sum()).groupby(level=1).sum()
        )
        if np.abs(firstStageShares.sum() - supply.sum()) > 1e-3:
            raise ValueError(
                "Total demand does not equal total supply! This may be a scaling issue."
            )
        if np.max(np.abs(firstStageShares - supply)) < convergence_criterion:
            break
        ascs = ascs - np.log(firstStageShares / supply)

        # normalize, can add/subtract constant to utility and not change predictions
        ascs -= ascs.iloc[0]

    return ascs
