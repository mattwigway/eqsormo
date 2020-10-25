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

# Author: Matthew Wigginton Conway <matt@indicatrix.org>, School of Geographical Sciences and Urban Planning, Arizona State University

import numpy as np
from logging import getLogger

LOG = getLogger(__name__)


# @numba.jit(nopython=True)
# TODO convergence_criterion doesn't do anything
def compute_ascs(
    base_utilities,
    supply,
    hhidx,
    choiceidx,
    starting_values=None,
    convergence_criterion=1e-6,
    weights=None,
    log=False,
):
    """
    Compute the alternative specific constants (ASCs) that should be added to base_utilities to make the market shares equal supply.

    This allows multinomial logit models with a full set of ASCs to converge a lot faster. For models with full sets of ASCs,
    the maximum-likelihood estimate will reproduce market shares perfectly. This means that the ASCs are implied by any set of coefficients.
    This algorithm finds those implied ASCs based on the "base" utilities provided - which are the utilities calculated based on the parameters,
    but without ASCs.

    This is based on equation 16 of Bayer et al (2004)

    :param base_utilities: Vector of utilities calculated using the current parameters and variable values, but no ASCs
    :type base_utilities: numpy.ndarray

    :param supply: supply of each alternative along each margin
    :type supply: list of numpy.ndarray

    :param hhidx: household index associated with each utility
    :type hhidx: numpy.ndarray

    :param choiceidx: housing type index associated with each utility
    :type choiceidx: numpy.ndarray

    :param starting_values: starting values for ASCs, default all zeros
    :type starting_values: list with numpy.ndarray with length equal to number of choices on each margin

    :param weights: household weights, indexed by hhidx (NOTE THAT SUPPLY SHOULD BE WEIGHTED AS WELL WHEN USING WEIGHTS)
    :type weights: numpy.ndarray
    """

    if starting_values is None:
        ascs = [np.zeros(chcidx.max() + 1) for chcidx in choiceidx]
    else:
        ascs = starting_values

    # this variable is unhelpfully named to increase efficiency, so that we do not have to allocate as many (potentially
    # large) arrays. at various points in the algorithm it contains utilities, then exponentiated utilities, then
    # probabilities
    values = np.copy(base_utilities)
    while True:
        values[:] = base_utilities
        # values now contains base utilities
        for margin in range(len(ascs)):
            np.add(values, ascs[margin][choiceidx[margin]], out=values)
        # values now contains full utilities with ASCs

        np.exp(values, out=values)
        # values now contains exponentiated utilities
        if np.any(~np.isfinite(values)):
            # TODO should raise ValueError, but that breaks numba
            print(
                "Some exponentiated utilities are non-finite! This may be a scaling issue."
            )
            # print('Max ASC:')
            # print(np.max(ascs))
            # print('Min ASC')
            # print(np.min(ascs))
            # print('NaNs:')
            # print(np.sum(np.isnan(ascs)))
            # print('Min utility')
            # print(np.min(first_stage_utilities))
            # print('Max utility')
            # print(np.max(first_stage_utilities))
            # print('Min exp(utility)')
            # print(np.min(exp_utils))
            # print('Max exp(utility)')
            # print(np.max(exp_utils))
            return [
                np.array([42.0])
            ]  # will cause errors somewhere else, so the process will crash, and hopefully the user
            # will find the output of the above print statement while debugging.

        logsums = np.bincount(hhidx, weights=values)
        np.divide(values, logsums[hhidx], out=values)
        # values now contains probabilities

        if weights is not None:
            # multiply each prob by the weight of the household it represents
            np.multiply(values, weights[hhidx], out=values)
        # values now contains weighted probabilities

        converged = True
        first_stage_shares = []

        # check convergence on all dimensions
        for margin in range(len(ascs)):
            margin_shares = np.bincount(choiceidx[margin], weights=values)
            first_stage_shares.append(margin_shares)

            if np.abs(np.sum(margin_shares) - np.sum(supply[margin])) > 1e-3:
                # TODO should raise ValueError, but that breaks numba
                print(
                    "Total demand does not equal total supply! This may be a scaling issue."
                )
                return [
                    np.array([42.0])
                ]  # will cause errors somewhere else, so the process will crash, and hopefully the user
                # will find the output of the above print statement while debugging.

            if not np.allclose(margin_shares, supply[margin]):
                converged = False
                if log:
                    LOG.info(
                        f"Margin {margin}: maxdiff: {np.max(margin_shares - supply[margin])}, mindiff: {np.min(margin_shares - supply[margin])} {len(ascs[margin])} ascs"
                    )

        if converged:
            break
        else:
            # update ASCs
            # this is not done in the above for loop because once the model has converged we don't want to update any more ASCs
            for margin in range(len(ascs)):
                ascs[margin] = ascs[margin] - np.log(
                    first_stage_shares[margin] / supply[margin]
                )

                # normalize, can add/subtract constant to utility and not change predictions
                # this is true even in the multidimensional case, because _every_ outcome changes simultaneously
                ascs[margin] -= ascs[margin][0]

    return ascs
