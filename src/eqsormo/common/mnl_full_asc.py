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
import scipy.optimize
import scipy.stats
import time
from logging import getLogger
import pandas as pd
import statsmodels.tools.numdiff

from .compute_ascs import compute_ascs
from .util import human_time

LOG = getLogger(__name__)


class MNLFullASC(object):
    """
    A multinomial logit model with full ASCs. With full ASCs, the model reproduces market shares perfectly. This
    can be exploited so that the maximum likelihood algorithm does not have to actually find the ASCs - for any set
    of coefficients, the ASCs that reproduce market shares perfectly are implied. This speeds estimation significantly.

    This class estimates such multinomial logit models.
    """

    def __init__(
        self,
        alternatives,
        supply,
        hhidx,
        choiceidx,
        chosen,
        starting_values,
        method="L-BFGS-B",
        asc_convergence_criterion=1e-6,
        param_names=None,
        est_ses=True,
        minimize_options={},
        add_ln_supply=(),
    ):
        """
        :param utility: Function that returns utility for all choice alternatives and decisionmakers for a given set of parameters, not including ASCs. Receives parameters as a numpy array. Should return a MultiIndexed-pandas dataframe with indices called 'decisionmaker' and 'choice'
        :type utility: function

        :param supply: List of supply of each choice in equilibrium, for each choice margin (choice margins may be simple or joint)
        :type supply: List of np.array

        :param choiceidx: List of arrays of what choice a particular utility is for, on each margin
        :type choiceidx: List of np.array

        :param starting_values: starting values for parameters
        :type starting_values: numpy array

        :param method: scipy optimize method, default 'bfgs'

        :param minimize_options: dict of additional options for selected minimization method
        :type minimize_options: dict

        :param add_ln_supply: margins (parallel to supply) that should have log of supply added to utility
        :type add_ln_supply: tuple
        """
        self.alternatives = (
            alternatives  # TODO will this always be called with self as first argument?
        )
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
        self.est_ses = est_ses
        self.minimize_options = minimize_options
        self.add_ln_supply = add_ln_supply

        # precompute the ln(supply) values to be added to utility
        self._log_supply = np.zeros_like(choiceidx[0], dtype="float64")
        for dim in add_ln_supply:
            self._log_supply += np.log(supply[dim][choiceidx[dim]])

    def base_utility(self, params):
        "Utility without ASCs"
        return np.dot(self.alternatives, params).reshape(-1) + self._log_supply

    def compute_ascs(self, base_utilities):
        """
        Compute alternative specific constants implied by base_utilities.

        Uses a contraction mapping found in equation 16 of Bayer et al 2004.
        """
        startTime = time.perf_counter()

        ascs = compute_ascs(
            base_utilities,
            self.supply,
            self.hhidx,
            self.choiceidx,
            starting_values=self._previous_ascs,
            convergence_criterion=self.asc_convergence_criterion,
        )
        self._previous_ascs = ascs  # speed convergence later

        endTime = time.perf_counter()
        totalTime = endTime - startTime
        # LOG.info(f'found ASCs in {human_time(totalTime)}')
        self.asc_time += totalTime
        return ascs

    def full_utility(self, params):
        "Full utilities including ASCs"
        base_utilities = self.base_utility(params)
        # LOG.info(f'Computed base utilities in {human_time(end_time - start_time)}')
        ascs = self.compute_ascs(base_utilities)
        full_utilities = base_utilities
        for margin in range(len(ascs)):
            # okay to use += here, base utilities will not be used again
            full_utilities += ascs[margin][self.choiceidx[margin]]
        return full_utilities

    def probabilities(self, utility):
        exp_utility = np.exp(utility)
        if not np.all(np.isfinite(exp_utility)):
            raise ValueError(
                "Household/choice combinations "
                + str(exp_utility.index[~np.isfinite(exp_utility)])
                + " have non-finite utilities!"
            )
        logsums = np.bincount(self.hhidx, weights=exp_utility)
        return exp_utility / logsums[self.hhidx]

    def choice_probabilities(self, utility):
        probabilities = self.probabilities(utility)
        return probabilities[self.chosen]

    def negative_log_likelihood_for_utility(self, utility):
        negll = -np.sum(np.log(self.choice_probabilities(utility)))
        if np.isnan(negll) or not np.isfinite(negll):
            LOG.warn(
                "log-likelihood nan or not finite"
            )  # just warn, solver may be able to get out of this
        # LOG.info(f'Current negative log likelihood: {negll:.2f}')
        return negll

    def negative_log_likelihood(self, params):
        return self.negative_log_likelihood_for_utility(self.full_utility(params))

    # some optimizers call with a second state arg, some do not. be lenient.
    def log_progress(self, params, state=None):
        negll = self.negative_log_likelihood(params)
        improvement = negll - self._prev_ll
        LOG.info(
            f"After iteration {self._iteration}, -ll {negll:.7f}, change: {improvement:.7f}"
        )
        self.negll_for_iteration.append(negll)
        self.params_for_iteration.append(np.copy(params))
        self._prev_ll = negll
        self._iteration += 1

    def fit(self):
        LOG.info("Fitting multinomial logit model")
        startTime = time.perf_counter()

        # this is in fact the log likelihood at constants, because all ASCs are still estimated
        # Note that this is only true if starting values are all zeros
        if not np.allclose(self.starting_values, 0):
            LOG.warn(
                "not all starting values are zero, log likelihood at constants is actually log likelihood at starting values and may be incorrect"
            )
        self.loglik_constants = -self.negative_log_likelihood(self.starting_values)

        self._iteration = 0
        self._prev_ll = -self.loglik_constants
        self.params_for_iteration = []
        self.negll_for_iteration = []
        LOG.info(f"Before iteration 0, -ll {self._prev_ll:.7f}")
        minResults = scipy.optimize.minimize(
            self.negative_log_likelihood,
            self.starting_values,
            method=self.method,
            options={"disp": True, **self.minimize_options},
            callback=self.log_progress,
        )
        self.params_for_iteration = np.array(self.params_for_iteration)
        self.negll_for_iteration = np.array(self.negll_for_iteration)

        if self.est_ses:
            if minResults.success:
                try:
                    LOG.info("calculating Hessian")
                    # TODO robust SEs
                    # hess = statsmodels.tools.numdiff.approx_hess3(minResults.x, self.negative_log_likelihood)
                    # hess = approx_hess3(minResults.x, self.negative_log_likelihood)
                    hess = self.linear_hessian(minResults.x)
                    LOG.info("inverting Hessian")
                    hessInv = np.linalg.inv(hess)
                    ses = np.sqrt(np.diag(hessInv))
                    LOG.info("done calculating and inverting Hessian")
                except KeyboardInterrupt:
                    self.est_ses = False
                    LOG.warn(
                        "Keyboard interrupt caught, not calculating standard errors"
                    )
            else:
                LOG.error("Not calculating Hessian because model did not converge")
                self.est_ses = False
        else:
            LOG.info(
                "Not calculating Hessian because standard errors were not requested"
            )

        self.loglik_beta = -self.negative_log_likelihood(minResults.x)
        if self.param_names is None:
            self.params = minResults.x
            if self.est_ses:
                self.se = ses
            else:
                self.se = None
        else:
            self.params = pd.Series(minResults.x, index=self.param_names)
            if self.est_ses:
                self.se = pd.Series(ses, index=self.param_names)
            else:
                self.se = None

        # TODO compute t-stats
        if self.est_ses:
            self.zvalues = self.params / self.se
            # two-tailed test
            self.pvalues = 2 * scipy.stats.norm.cdf(-np.abs(self.zvalues))

        self.ascs = self.compute_ascs(self.base_utility(minResults.x))
        self.converged = minResults.success

        endTime = time.perf_counter()
        if self.converged:
            LOG.info(
                f"Multinomial logit model converged in {human_time(endTime - startTime)}: {minResults.message}"
            )
        else:
            LOG.error(
                f"Multinomial logit model FAILED TO CONVERGE in {human_time(endTime - startTime)}: {minResults.message}"
            )
        LOG.info(f"  Finding ASCs took {human_time(self.asc_time)}")

    def linear_hessian(self, params, step_size=None):
        """
        Compute the Hessian, quickly. This takes advantage of the fact that many components of utility are constant
        when computing the Hessian, which changes two parameters at a time. The ASCs also do not really need to be
        recomputed, as the changes to the coefficients are miniscule. This only works for utility functions that are
        strictly linear-in-parameters, which the MNL utility function is.

        The results of this function should be identical to statsmodels.tools.numdiff.approx_hess3, as the algorithm
        used is the same (equation 9 from Ridout 2009 - note that Ridout 2009 is about complex step differentiation, but
        eq. 9 is an example of finite-step differentiation).

        Ridout, M. S. (2009). Statistical Applications of the Complex-Step Method of Numerical Differentiation.
            _The American Statistician_, 63(1), 66–74. https://doi.org/10.1198/tast.2009.0013
        """
        utility = self.full_utility(params)

        # this is the step size for the finite-difference approximation. it is copied from
        # statsmodels.tools.numdiff.approx_hess3
        if step_size is None:
            step_size = np.MachAr().eps ** (1.0 / 4) * np.maximum(np.abs(params), 0.1)
        elif isinstance(step_size, float):
            step_size = np.full(len(params), step_size)

        LOG.info("computing Hessian")
        # we need the outer product to scale the finite diff appx below
        prod = np.outer(step_size, step_size)
        hess = np.zeros_like(prod)

        n = len(params)
        for i in range(n):
            step_i = self.alternatives[:, i] * step_size[i]
            LOG.info(f"Hessian starting row {i} / {n}")
            for j in range(i, n):
                step_j = self.alternatives[:, j] * step_size[j]
                hess[i, j] = hess[j, i] = (
                    self.negative_log_likelihood_for_utility(utility + step_i + step_j)
                    - self.negative_log_likelihood_for_utility(
                        utility + step_i - step_j
                    )
                    - (
                        self.negative_log_likelihood_for_utility(
                            utility - step_i + step_j
                        )
                        - self.negative_log_likelihood_for_utility(
                            utility - step_i - step_j
                        )
                    )
                ) / (4 * prod[i, j])

        return hess

    def summary(self):
        with pd.option_context("display.max_rows", None):
            notes = []
            if self.est_ses:
                summ = pd.DataFrame(
                    {
                        "coef": self.params,
                        "se": self.se,
                        "z": self.zvalues,
                        "p": self.pvalues,
                    }
                ).round(3)
            else:
                summ = pd.DataFrame({"coef": self.params}).round(3)
                notes.append("Standard errors not estimated")

            if not self.converged:
                notes.append("WARNING: CONVERGENCE NOT ACHIEVED.")

            chi2 = 2 * (self.loglik_beta - self.loglik_constants)
            chi2df = len(self.params)
            pchi2 = 1 - scipy.stats.chi2.cdf(chi2, df=chi2df)

            # get around f-string cannot include backslash error
            newline = "\n"

            return f"""
Multinomial logit model with full ASCs
Parameters:
{str(summ)}
Notes:
{newline.join(notes)}

Log likelihood at constants: {self.loglik_constants:.3f}
Log likelihood at convergence: {self.loglik_beta:.3f}
Pseudo R-squared (McFadden): {1 - self.loglik_beta / self.loglik_constants:.3f}
Chi2 (vs constants): {chi2}, {chi2df} degrees of freedom
P(Chi2): {pchi2}
            """

        # don't pickle big arrays
        def __getstate__(self):
            return {
                k: v
                for k, v in self.__dict__.items()
                if k not in {"hhidx", "choiceidx", "chosen"}
            }
