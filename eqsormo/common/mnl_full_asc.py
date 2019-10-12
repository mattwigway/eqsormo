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

import numpy as np
import scipy.optimize
import time
from logging import getLogger
import pandas as pd
import time

LOG = getLogger(__name__)

class MNLFullASC(object):
    '''
    A multinomial logit model with full ASCs. With full ASCs, the model reproduces market shares perfectly. This
    can be exploited so that the maximum likelihood algorithm does not have to actually find the ASCs - for any set
    of coefficients, the ASCs that reproduce market shares perfectly are implied. This speeds estimation significantly.

    This class estimates such multinomial logit models.
    '''

    def __init__ (self, utility, choice, supply, starting_values, method='bfgs', asc_convergence_criterion=1e-6, param_names=None):
        '''
        :param utility: Function that returns utility for all choice alternatives and decisionmakers for a given set of parameters, not including ASCs. Receives parameters as a numpy array. Should return a MultiIndexed-pandas dataframe with indices called 'decisionmaker' and 'choice'
        :type utility: function

        :param supply: Supply of each choice in equilibrium
        :type supply: Pandas series, indexed with choice alternatives (like return value of utility function)

        :param starting_values: starting values for parameters
        :type starting_values: numpy array

        :param method: scipy optimize method, default 'bfgs'
        '''
        self.utility = utility # TODO will this always be called with self as first argument?
        self.supply = supply
        self.starting_values = starting_values
        self.method = method
        self.asc_convergence_criterion = asc_convergence_criterion
        self.choice = choice
        self.param_names = param_names
        self._previous_ascs = None
        self.asc_time = 0

    def compute_ascs (self, baseUtilities, params):
        '''
        Compute alternative specific constants implied by params.

        Uses a contraction mapping found in equation 16 of Bayer et al 2004.
        '''
        startTime = time.clock()
        if self._previous_ascs is not None:
            ascs = self._previous_ascs
        else:
            uniqueChoices = baseUtilities.index.unique(level='choice')
            ascs = pd.Series(np.zeros(len(uniqueChoices)), index=uniqueChoices)

        while True:
            firstStageUtilities = baseUtilities + ascs.loc[baseUtilities.index.get_level_values('choice')].values
            expUtils = np.exp(firstStageUtilities)
            if np.any(~np.isfinite(expUtils)):
                if self.param_names is not None:
                    params = pd.Series(params, index=self.param_names)
                raise ValueError('Some utilities are non-finite! This may be a scaling issue.\n' +\
                    f'Current values of the parameters: {params}')
            firstStageShares = (expUtils / expUtils.groupby(level=0).sum()).groupby(level=1).sum()
            if np.abs(firstStageShares.sum() - self.supply.sum()) > 1e-3:
                raise ValueError('Total demand does not equal total supply! This may be a scaling issue.')
            if np.max(np.abs(firstStageShares - self.supply)) < self.asc_convergence_criterion:
                break
            ascs = ascs - np.log(firstStageShares / self.supply)

            # normalize, can add/subtract constant to utility and not change predictions
            ascs -= ascs.iloc[0]

        self._previous_ascs = ascs # speed convergence later

        endTime = time.clock()
        self.asc_time += (endTime - startTime)
        return ascs

    def full_utility (self, params):
        'Full utilities including ASCs'
        baseUtilities = self.utility(params)
        ascs = self.compute_ascs(baseUtilities, params)
        fullUtilities = baseUtilities + ascs.loc[baseUtilities.index.get_level_values('choice')].values
        return fullUtilities

    def probabilities (self, params):
        utility = self.full_utility(params)
        expUtility = np.exp(utility)
        if not np.all(np.isfinite(expUtility)):
            raise ValueError(f'Household/choice combinations {expUtility.index[~np.isfinite(expUtility)]} have non-finite utilities!')
        return expUtility / expUtility.groupby(level=0).sum()

    def choice_probabilities (self, params):
        probabilities = self.probabilities(params)
        return probabilities.loc[list(zip(self.choice.index, self.choice))]

    def negative_log_likelihood (self, params):
        negll = -np.sum(np.log(self.choice_probabilities(params)))
        if np.isnan(negll) or not np.isfinite(negll):
            if self.param_names is not None:
                params = pd.Series(params, index=self.param_names)
            LOG.warn(f'log-likelihood nan or not finite, for params:\n{params}') # just warn, solver may be able to get out of this
        return negll

    def fit (self):
        LOG.info('Fitting multinomial logit model')
        startTime = time.clock()

        # this is in fact the log likelihood at constants, because all ASCs are still estimated
        self.loglik_constants = -self.negative_log_likelihood(np.zeros_like(self.starting_values))

        minResultsInitial = scipy.optimize.minimize(
            self.negative_log_likelihood,
            self.starting_values,
            method=self.method,
            options={'disp': True}
        )

        if self.method != 'bfgs':
            # run one iter of bfgs to get SEs
            minResultsFinal = scipy.optimize.minimize(
                self.negative_log_likelihood,
                minResultsInitial.x,
                method='bfgs'
            )
        else:
            minResultsFinal = minResultsInitial

        self.loglik_beta = -self.negative_log_likelihood(minResultsFinal.x)
        if self.param_names is None:
            self.params = minResultsFinal.x
            self.se = np.sqrt(np.diag(minResultsFinal.hess_inv))
        else:
            self.params = pd.Series(minResultsFinal.x, index=self.param_names)
            self.se = pd.Series(np.sqrt(np.diag(minResultsFinal.hess_inv)), index=self.param_names)

        # TODO compute t-stats
        self.zvalues = self.params / self.se
        self.pvalues = scipy.stats.norm.cdf(1 - np.abs(self.zvalues))

        # TODO robust SEs
        self.ascs = self.compute_ascs(self.utility(minResultsFinal.x), minResultsFinal.x)
        self.converged = minResultsInitial.success

        endTime = time.clock()
        if self.converged:
            LOG.info(f'Multinomial logit model converged in {endTime - startTime:.3f} seconds: {minResultsInitial.message}')
        else:
            LOG.error(f'Multinomial logit model FAILED TO CONVERGE in {endTime - startTime:.3f} seconds: {minResultsInitial.message}')
        LOG.info(f'  Finding ASCs took {self.asc_time:.3f} seconds')


    def summary (self):
        summ = pd.DataFrame({
            'coef': self.params,
            'se': self.se,
            'z': self.zvalues,
            'p': self.pvalues
        }).round(3)
        return f'''
Multinomial logit model with full ASCs
Parameters:
{str(summ)}

Log likelihood at constants: {self.loglik_constants:.3f}
Log likelihood at convergence: {self.loglik_beta:.3f}
Pseudo R-squared (McFadden): {1 - self.loglik_beta / self.loglik_constants:.3f}
        '''

