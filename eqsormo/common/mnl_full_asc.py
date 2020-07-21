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
import scipy.optimize, scipy.stats
import time
from logging import getLogger
import pandas as pd
import statsmodels.tools.numdiff

from .compute_ascs import compute_ascs
from .util import human_time

LOG = getLogger(__name__)

class MNLFullASC(object):
    '''
    A multinomial logit model with full ASCs. With full ASCs, the model reproduces market shares perfectly. This
    can be exploited so that the maximum likelihood algorithm does not have to actually find the ASCs - for any set
    of coefficients, the ASCs that reproduce market shares perfectly are implied. This speeds estimation significantly.

    This class estimates such multinomial logit models.
    '''

    def __init__ (self, utility, supply, hhidx, choiceidx, chosen, starting_values, method='L-BFGS-B', asc_convergence_criterion=1e-6, param_names=None):
        '''
        :param utility: Function that returns utility for all choice alternatives and decisionmakers for a given set of parameters, not including ASCs. Receives parameters as a numpy array. Should return a MultiIndexed-pandas dataframe with indices called 'decisionmaker' and 'choice'
        :type utility: function

        :param supply: List of supply of each choice in equilibrium, for each choice margin (choice margins may be simple or joint)
        :type supply: List of np.array

        :param choiceidx: List of arrays of what choice a particular utility is for, on each margin
        :type choiceidx: List of np.array

        :param starting_values: starting values for parameters
        :type starting_values: numpy array

        :param method: scipy optimize method, default 'bfgs'
        '''
        self.utility = utility # TODO will this always be called with self as first argument?
        self.supply = supply
        self.starting_values = starting_values
        self.method = method
        self.asc_convergence_criterion = asc_convergence_criterion
        self.param_names = param_names
        self._previous_ascs = None
        self.asc_time = 0
        self.hhidx = hhidx
        self.choiceidx = choiceidx
        self.chosen = chosen

    def compute_ascs (self, base_utilities):
        '''
        Compute alternative specific constants implied by base_utilities.

        Uses a contraction mapping found in equation 16 of Bayer et al 2004.
        '''
        startTime = time.perf_counter()

        ascs = compute_ascs(base_utilities, self.supply, self.hhidx, self.choiceidx, starting_values=self._previous_ascs, convergence_criterion=self.asc_convergence_criterion)
        self._previous_ascs = ascs # speed convergence later

        endTime = time.perf_counter()
        totalTime = endTime - startTime
        #LOG.info(f'found ASCs in {human_time(totalTime)}')
        self.asc_time += totalTime
        return ascs

    def full_utility (self, params):
        'Full utilities including ASCs'
        start_time = time.perf_counter()
        base_utilities = self.utility(params)
        end_time = time.perf_counter()
        #LOG.info(f'Computed base utilities in {human_time(end_time - start_time)}')
        ascs = self.compute_ascs(base_utilities)
        full_utilities = base_utilities
        for margin in range(len(ascs)):
            # okay to use += here, base utilities will not be used again
            full_utilities += ascs[margin][self.choiceidx[margin]]
        return full_utilities

    def probabilities (self, params):
        utility = self.full_utility(params)
        exp_utility = np.exp(utility)
        if not np.all(np.isfinite(exp_utility)):
            raise ValueError('Household/choice combinations ' + str(exp_utility.index[~np.isfinite(exp_utility)]) + ' have non-finite utilities!')
        logsums = np.bincount(self.hhidx, weights=exp_utility)
        return exp_utility / logsums[self.hhidx]

    def choice_probabilities (self, params):
        probabilities = self.probabilities(params)
        return probabilities[self.chosen]

    def negative_log_likelihood (self, params):
        negll = -np.sum(np.log(self.choice_probabilities(params)))
        if np.isnan(negll) or not np.isfinite(negll):
            LOG.warn('log-likelihood nan or not finite, for params:\n' + str(params)) # just warn, solver may be able to get out of this
        #LOG.info(f'Current negative log likelihood: {negll:.2f}')
        return negll

    # some optimizers call with a second state arg, some do not. be lenient.
    def log_progress (self, params, state=None):
        negll = self.negative_log_likelihood(params)
        improvement = negll - self._prev_ll
        LOG.info(f'After iteration {self._iteration}, -ll {negll:.7f}, change: {improvement:.7f}')
        self.negll_for_iteration.append(negll)
        self.params_for_iteration.append(np.copy(params))
        self._prev_ll = negll
        self._iteration += 1

    def fit (self):
        LOG.info('Fitting multinomial logit model')
        startTime = time.perf_counter()

        # this is in fact the log likelihood at constants, because all ASCs are still estimated
        # Note that this is only true if starting values are all zeros
        if not np.allclose(self.starting_values, 0):
            LOG.warn('not all starting values are zero, log likelihood at constants is actually log likelihood at starting values and may be incorrect')
        self.loglik_constants = -self.negative_log_likelihood(self.starting_values)

        self._iteration = 0
        self._prev_ll = -self.loglik_constants
        self.params_for_iteration = []
        self.negll_for_iteration = []
        minResults = scipy.optimize.minimize(
            self.negative_log_likelihood,
            self.starting_values,
            method=self.method,
            options={'disp': True},
            callback=self.log_progress
        )
        self.params_for_iteration = np.array(self.params_for_iteration)
        self.negll_for_iteration = np.array(self.negll_for_iteration)

        LOG.info('calculating and inverting Hessian')
        hess = statsmodels.tools.numdiff.approx_hess3(minResults.x, self.negative_log_likelihood)
        hessInv = np.linalg.inv(hess)
        ses = np.sqrt(np.diag(hessInv))
        LOG.info('done calculating and inverting Hessian')

        self.loglik_beta = -self.negative_log_likelihood(minResults.x)
        if self.param_names is None:
            self.params = minResults.x
            self.se = ses
        else:
            self.params = pd.Series(minResults.x, index=self.param_names)
            self.se = pd.Series(ses, index=self.param_names)

        # TODO compute t-stats
        self.zvalues = self.params / self.se
        self.pvalues = scipy.stats.norm.cdf(1 - np.abs(self.zvalues))

        # TODO robust SEs
        self.ascs = self.compute_ascs(self.utility(minResults.x))
        self.converged = minResults.success

        endTime = time.perf_counter()
        if self.converged:
            LOG.info(f'Multinomial logit model converged in {endTime - startTime:.3f} seconds: {minResults.message}')
        else:
            LOG.error(f'Multinomial logit model FAILED TO CONVERGE in {endTime - startTime:.3f} seconds: {minResults.message}')
        LOG.info(f'  Finding ASCs took {self.asc_time:.3f} seconds')

    def summary (self):
        summ = pd.DataFrame({
            'coef': self.params,
            'se': self.se,
            'z': self.zvalues,
            'p': self.pvalues
        }).round(3)

        chi2 = 2 * (self.loglik_beta - self.loglik_constants)
        chi2df = len(self.params)
        pchi2 = 1 - scipy.stats.chi2.cdf(chi2, df=chi2df)

        return f'''
Multinomial logit model with full ASCs
Parameters:
{str(summ)}

Log likelihood at constants: {self.loglik_constants:.3f}
Log likelihood at convergence: {self.loglik_beta:.3f}
Pseudo R-squared (McFadden): {1 - self.loglik_beta / self.loglik_constants:.3f}
Chi2 (vs constants): {chi2}, {chi2df} degrees of freedom
P(Chi2): {pchi2}
        '''

