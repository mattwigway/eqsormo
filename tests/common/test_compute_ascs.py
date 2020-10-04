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

# Author: Matthew Wigginton Conway <matt@indicatrix.org>
#         School of Geographical Sciences and Urban Planning
#         Arizona State University

import numpy as np
from eqsormo.common.compute_ascs import compute_ascs


def test_compute_ascs ():
    # suppose we have 4 decisionmakers and a 4x4 grid of alternatives, so 4 * 4 * 4 items
    utilities = np.array([14.52, 2.61, 15.36, 4.46, 0.41, 19.45, 11.01, 9.95, 3.05,
                          6.62, 15.56, 17.75, 10.54, 12.77, 6.04, 2.77, 1.27, 3.34,
                          14.99, 1.23, 6.3, 4.11, 17.54, 3.13, 18.31, 5.62, 3.89,
                          1.48, 16.81, 12.32, 12.96, 11.51, 8.51, 8.44, 15.68, 2.55,
                          7.79, 5.34, 14.84, 10.41, 1.97, 17.71, 5.15, 13.43, 14.91,
                          18.4, 0.22, 10.56, 16.08, 0.6, 0.57, 18.07, 9.63, 4.32,
                          0.96, 3.23, 9.98, 5.74, 0.07, 13.94, 7.8, 15.39, 1.73,
                          4.96])

    hhidx = np.repeat(np.arange(4), 16)
    choiceidx = [
        np.tile(np.repeat(np.arange(4), 4), 4),
        np.tile(np.arange(4), 16)
    ]

    supply = [
        # has to sum to 4
        np.array([1.5, 0.75, 1, 0.75]),
        np.array([1.25, 1.5, 0.5, 0.75])
    ]

    ascs = compute_ascs(
        base_utilities=utilities,
        supply=supply,
        hhidx=hhidx,
        choiceidx=choiceidx
    )

    # compute probabilities
    full_exp_utilities = np.exp(utilities + ascs[0][choiceidx[0]] + ascs[1][choiceidx[1]])

    full_sums = np.bincount(hhidx, weights=full_exp_utilities)
    probs = full_exp_utilities / full_sums[hhidx]
    shares_0 = np.bincount(choiceidx[0], weights=probs)
    shares_1 = np.bincount(choiceidx[1], weights=probs)

    assert np.allclose(shares_0, supply[0]), 'ascs do not produce correct market shares for dim 0'
    assert np.allclose(shares_1, supply[1]), 'ascs do not produce correct market shares for dim 1'


def test_compute_ascs_weights ():
    # suppose we have 4 decisionmakers and a 4x4 grid of alternatives, so 4 * 4 * 4 items
    utilities = np.array([14.52, 2.61, 15.36, 4.46, 0.41, 19.45, 11.01, 9.95, 3.05,
                          6.62, 15.56, 17.75, 10.54, 12.77, 6.04, 2.77, 1.27, 3.34,
                          14.99, 1.23, 6.3, 4.11, 17.54, 3.13, 18.31, 5.62, 3.89,
                          1.48, 16.81, 12.32, 12.96, 11.51, 8.51, 8.44, 15.68, 2.55,
                          7.79, 5.34, 14.84, 10.41, 1.97, 17.71, 5.15, 13.43, 14.91,
                          18.4, 0.22, 10.56, 16.08, 0.6, 0.57, 18.07, 9.63, 4.32,
                          0.96, 3.23, 9.98, 5.74, 0.07, 13.94, 7.8, 15.39, 1.73,
                          4.96])

    hhidx = np.repeat(np.arange(4), 16)
    choiceidx = [
        np.tile(np.repeat(np.arange(4), 4), 4),
        np.tile(np.arange(4), 16)
    ]

    weights = np.array([5, 2, 8, 3])

    supply = [
        # has to sum to 18, like weights
        np.array([6, 4, 3, 5]),
        np.array([5, 2, 7, 4])
    ]

    ascs = compute_ascs(
        base_utilities=utilities,
        supply=supply,
        hhidx=hhidx,
        choiceidx=choiceidx,
        weights=weights
    )

    # compute probabilities
    full_exp_utilities = np.exp(utilities + ascs[0][choiceidx[0]] + ascs[1][choiceidx[1]])

    full_sums = np.bincount(hhidx, weights=full_exp_utilities)
    probs = full_exp_utilities / full_sums[hhidx]
    probs *= weights[hhidx]
    shares_0 = np.bincount(choiceidx[0], weights=probs)
    shares_1 = np.bincount(choiceidx[1], weights=probs)

    assert np.allclose(shares_0, supply[0]), 'ascs do not produce correct market shares for dim 0'
    assert np.allclose(shares_1, supply[1]), 'ascs do not produce correct market shares for dim 1'
