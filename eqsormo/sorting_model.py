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

from logging import getLogger
import time
import statsmodels.api as sm
import pandas as pd
import numpy as np
import scipy.optimize, scipy.stats
import linearmodels
from tqdm import tqdm

LOG = getLogger(__name__)

class SortingModel(object):
    """
    Represents a random utility based equilibrium sorting model.

    Based on the model described in Klaiber and Phaneuf (2010) "Valuing open space in a residential sorting model of the Twin Cities"
    https://doi.org/10.1016/j.jeem.2010.05.002 (no open access version available, unfortunately)
    """

    def __init__ (self, altHousing, altNeighborhood, altPrice, altHedonic, hh, hhChoice, interactions, initialPriceCoef, sampleAlternatives=None, method='bfgs'):
        """
        :param altHousing: Home attributes of housing alternatives/types (assumed that all alternatives are available to all households)
        :type altHousing: Pandas dataframe

        :param altNeighborhood: Exogenous neighborhood attributes of housing alternatives (endogenous neighborhood attributes not yet supported). Column names must be different from those in altHousing.
        :type altNeighborhood: Pandas dataframe, indexed like altHousing
        
        :param altPrice: Equilibrium (non-instrumented) price of housing alternatives (will be replaced with instrumented version)
        :type altPrice: Pandas series of float64, indexed like altHousing and altNeighborhood

        :param altHedonic: Attributes of alternatives to be used in hedonic estimation (i.e. instrumenting for price). Will likely be a superset of altHousing and altNeighborhood, with some other attributes about nearby neighborhoods
        :type altHedonic: Pandas dataframe, indexed like altHousing, altNeighborhood, and altPrice

        :param hh: Attributes of households
        :type hh: Pandas dataframe

        :param hhChoice: Alternative actually chosen by households
        :type hhChoice: Pandas series, indexed like hh, with values matching index of altHousing et al.

        :param interactions: Which sociodemographics should be interacted with which housing or neighborhood attributes.
        :type interactions: iterable of tuple (sociodemographic column, housing/neighborhood column)

        :param initialPriceCoef: For the 2SLS estimate, an initial estimate for the price coefficient is needed; this value is iterated from and should converge.
        :type intialPriceCoef: float

        :param sampleAlternatives: If a positive integer, take a random sample of this size for available alternatives to estimate the model. Recommended in models with a large number of alternatives. If 0 or None, no sampling will be performed. Use np.set_seed() before calling fit for model reproducibility when using this option.
        :type sampleAlternatives: int or None

        :param method: A scipy.optimize.minimize method used for maximizing log-likelihood of first stage, default 'bfgs'
        """

        # Protective copies of all the things
        self.altHousing = altHousing.copy()
        self.altNeighborhood = altNeighborhood.copy()
        self.price = altPrice.copy()
        self.altHedonic = altHedonic.copy()
        self.hh = hh.copy().apply(lambda col: col - col.mean()) # predemean household characteristics
        self.hh.index.name = 'household' # for joining convenience later
        self.hhChoice = hhChoice.copy()
        self.interactions = interactions
        self.initialPriceCoef = initialPriceCoef
        self.sampleAlternatives = sampleAlternatives

        #: supply by housing type
        self.supply = self.hhChoice.value_counts().astype('float64')

        self.validate()

        self.altCharacteristics = self.altHousing.join(self.altNeighborhood)
        assert not 'price' in self.altCharacteristics.columns
        self.altCharacteristics['price'] = altPrice
        self.method = method

        self.MARKET_CLEARING_TOLERANCE = 1e-2 # how far from the true supply can the market share be (in this case, within 0.01 units)

    def validate (self):
        "Check for obvious errors in model inputs"
        LOG.warn('Validation function not implemented. You better be darn sure of your inputs.')

    def fit (self):
        'Fit the whole model'
        startTime = time.clock()
        LOG.info('Fitting equilibrium sorting model')
        if self.alternatives is None:
            raise RuntimeError('Alternatives not yet created')

        self.fit_first_stage()
        self.fit_second_stage()
        endTime = time.clock()
        LOG.info(f'''
Fitting sorting model took {endTime - startTime:.3f} seconds.
Convergence:
  First stage: {self.first_stage_converged}
''')
    def create_alternatives (self):
        LOG.info('Creating alternatives')
        startTime = time.clock()

        self.fullAlternatives = pd.concat([self.altCharacteristics for i in range(len(self.hh))], keys=self.hh.index)
        self.fullAlternatives['chosen'] = False
        self.fullAlternatives['hhchoice'] = self.hhChoice.reindex(self.fullAlternatives.index, level=0)
        self.fullAlternatives.loc[self.fullAlternatives.index.get_level_values(1) == self.fullAlternatives.hhchoice, 'chosen'] = True

        LOG.info('created full set of alternatives, now sampling if requested')

        if self.sampleAlternatives <= 0 or self.sampleAlternatives is None:
            self.alternatives = self.fullAlternatives
        else:
            unchosenAlternatives = self.fullAlternatives[~self.fullAlternatives.chosen].groupby(level=0).apply(lambda x: x.sample(self.sampleAlternatives - 1))
            unchosenAlternatives.index = unchosenAlternatives.index.droplevel(0) # fix dup'd household level due to groupby
            self.alternatives = pd.concat([unchosenAlternatives, self.fullAlternatives[self.fullAlternatives.chosen]]).sort_index(level=[0, 1])

        self.alternatives.drop(columns=['chosen'], inplace=True)
        self.fullAlternatives.drop(columns=['chosen'], inplace=True)

        endTime = time.clock()
        LOG.info(f'Created {len(self.alternatives)} alternatives for {len(self.hh)} in {endTime - startTime:.3f} seconds')

    def first_stage_utility (self, params, mean_indirect_utility):
        # TODO I don't think that adding the mean_indirect_utility this way will work from an indexing standpoint
        # diffs is differences due to sociodemographics from mean indirect utility
        diffs = self.firstStageData.multiply(params, axis='columns').sum(axis='columns')
        return diffs + mean_indirect_utility.loc[self.firstStageData.index.get_level_values('choice')].values

    def first_stage_logprobabilities (self, params, mean_indirect_utility):
        utility = self.first_stage_utility(params, mean_indirect_utility)
        expUtility = np.exp(utility)
        if not np.all(np.isfinite(expUtility)):
            raise ValueError(f'Household/choice combinations {expUtility.index[~np.isfinite(expUtility)]} have non-finite utilities!')
        return utility - np.log(expUtility.groupby(level=0).sum())

    def compute_mean_indirect_utility (self, params):
        # These are the alternative specific constants, which are not fit by ML, but rather using a contraction mapping that lets
        # the model converge faster - see Equation 16 of Bayer et al. (2004).
        # TODO better starting values for this?
        LOG.debug('Computing mean indirect utilities (ASCs)')
        startTime = time.clock()
        mean_indirect_utility = self._prev_mean_indirect_utility
        probs = np.exp(self.first_stage_logprobabilities(params, mean_indirect_utility)).groupby(level=1).sum() # group by housing types
        # optimization: save starting values (makes it converge faster next time)
        iter = 0
        while True:
            iter += 1
            mean_indirect_utility = mean_indirect_utility - np.log(probs / self.supply)
            probs = np.exp(self.first_stage_logprobabilities(params, mean_indirect_utility)).groupby(level=1).sum() # group by housing types
            # TODO hardcoded tolerances below are bad
            if np.abs(probs.sum() - self.supply.sum()) > 1e-3:
                raise ValueError('Total demand does not equal total supply! This may be a scaling issue.')
            if np.max(np.abs(probs - self.supply)) < 1e-6:
                break
        
        endTime = time.clock()
        LOG.debug(f'Computed {len(mean_indirect_utility)} mean indirect utilities in {endTime-startTime:.3f}s using {iter} iterations')

        self._prev_mean_indirect_utility = mean_indirect_utility
        return mean_indirect_utility

    def first_stage_neg_loglikelihood (self, params):
        mean_indirect_utility = self.compute_mean_indirect_utility(params)
        logprobs = self.first_stage_logprobabilities(params, mean_indirect_utility)
        return -np.sum(logprobs.loc[list(zip(self.hhChoice.index, self.hhChoice))])

    def fit_first_stage (self):
        'Perform the first stage estimation'
        LOG.info('Performing first-stage estimation')

        startTime = time.clock()

        self.firstStageData = pd.DataFrame()

        # demean sociodemographics so ASCs are interpretable as mean indirect utility (or something like that... TODO check)
        altsWithHhCharacteristics = self.alternatives.join(self.hh) # should project to all alternatives TODO check

        for interaction in self.interactions:
            self.firstStageData[f'{interaction[0]}_{interaction[1]}'] =\
                 altsWithHhCharacteristics[interaction[0]] * altsWithHhCharacteristics[interaction[1]]
            # solve scaling issues
            stdevs = self.firstStageData.apply(np.std)
            self.firstStageData = self.firstStageData.divide(stdevs, axis='columns')
        
        LOG.info(f'Fitting {len(self.firstStageData.columns)} interaction parameters')

        self._prev_mean_indirect_utility = pd.Series(np.zeros(len(self.altCharacteristics)), index=self.altCharacteristics.index)

        self.first_stage_loglik_constants = -self.first_stage_neg_loglikelihood(np.zeros(len(self.firstStageData.columns)))

        minResults = scipy.optimize.minimize(
            self.first_stage_neg_loglikelihood,
            np.zeros(len(self.firstStageData.columns)),
            method=self.method,
            options={'disp': True}
        )

        self.first_stage_loglik_beta = -self.first_stage_neg_loglikelihood(minResults.x)
        self.interaction_params = pd.Series(minResults.x, self.firstStageData.columns) * stdevs # correct the scaling
        self.interaction_params_se = pd.Series(np.sqrt(np.diag(minResults.hess_inv)), self.firstStageData.columns) * stdevs
        # TODO robust SEs
        self.mean_indirect_utility = self.compute_mean_indirect_utility(self.interaction_params)
        self.first_stage_converged = minResults.success

        endTime = time.clock()
        if self.first_stage_converged:
            LOG.info(f'First stage converged in {endTime - startTime:.3f} seconds: {minResults.message}')
        else:
            LOG.error(f'First stage FAILED TO CONVERGE in {endTime - startTime:.3f} seconds: {minResults.message}')

    def fit_second_stage (self):
        'Fit the instrumental variables portion of the model'
        LOG.info('Fitting second stage')

        startTime = time.clock()

        priceCoef = prevPriceCoef = self.initialPriceCoef

        alts = self.altHousing.join(self.altNeighborhood)

        iter = 0
        with tqdm() as pbar:
            while True:
                residual_utility = self.mean_indirect_utility - priceCoef * self.price
                # Constant should be included since location of ASCs is arbitrary, see Klaiber and Kuminoff (2014) note 9
                olsreg = sm.OLS(residual_utility, sm.add_constant(alts))
                olsfit = olsreg.fit()

                # compute the price instrument by solving for prices that clear the market - which means the prices that produce the mean indirect utilities found
                # in the first stage, since those are the market-clearing utilities.
                priceIv = (self.mean_indirect_utility - olsfit.fittedvalues) / priceCoef

                ivreg = linearmodels.IV2SLS(self.mean_indirect_utility, sm.add_constant(alts), pd.DataFrame(self.price.rename('price')), pd.DataFrame(priceIv.rename('price_iv')))
                self.second_stage_fit = ivreg.fit()

                priceCoef = self.second_stage_fit.params.price

                pbar.update()
                if np.abs(priceCoef - prevPriceCoef) < 1e-6:
                    endTime = time.clock()
                    LOG.info(f'Price coefficient converged in {endTime-startTime:.3f} seconds after {iter} iterations')
                    self.mean_params = self.second_stage_fit.params
                    #self.mean_params_se = self.second_stage_fit.bse
                    self.type_shock = self.second_stage_fit.resids
                    self.price_iv = priceIv
                    break
                else:
                    prevPriceCoef = priceCoef

    def summary (self):
        # summarize params
        summary = pd.DataFrame({
            'coef': pd.concat([self.interaction_params, self.mean_params]),
            'se': self.interaction_params_se # TODO no standard errors for second stage yet...
        })

        summary['z'] = summary.coef / summary.se
        # TODO should probably be t-test, but then I have to do a bunch of df calculations...
        summary['p'] = (1 - scipy.stats.norm.cdf(np.abs(summary.z))) * 2
        return summary

    def utilities (self):
        "Get utilities of every household choosing every house type"
        fullAlternativesWithInteractions = self.fullAlternatives.join(self.hh)

        for left, right in self.interactions:
            fullAlternativesWithInteractions[f'{left}_{right}'] = fullAlternativesWithInteractions[left] * fullAlternativesWithInteractions[right]

        fullAlternativesWithInteractions['price'] = self.price.reindex(fullAlternativesWithInteractions.index, level=1)

        # compute utilities, without unobserved price shocks
        params = pd.concat([self.mean_params, self.interaction_params])
        utilities = sm.add_constant(fullAlternativesWithInteractions)[params.index].multiply(params, 'columns').sum(axis='columns')
        utilities += self.type_shock.reindex(fullAlternativesWithInteractions.index, level=1)

        return utilities

    def probabilities (self):
        expUtilities = np.exp(self.utilities())
        logsums = expUtilities.groupby(level=0).sum() # group by households and sum
        probabilities = expUtilities / logsums.reindex(expUtilities.index, level=0)
        return probabilities

    def market_shares (self):
        return self.probabilities().groupby(level=1).sum()

    def clear_market (self):
        'Adjust prices so the market clears'
        prices = self.price # for comparison later

        LOG.info('Clearing the market (everyone stand back)')

        mktShares = self.market_shares()
        iters = 0
        with tqdm() as pbar:
            while np.max(np.abs(mktShares - self.supply)) > self.MARKET_CLEARING_TOLERANCE:
                self.price *= (((mktShares / self.supply) - 1) * -self.mean_params.price) + 1
                mktShares = self.market_shares()
                iters += 1
                pbar.update(1)
                pbar.set_description(f'max abs diff: {np.max(np.abs(mktShares - self.supply)):.3f} units')

        maxPriceChange = np.max(np.abs(self.price - prices))
        LOG.info(f'Market cleared after {iters} iterations, with absolute price changes of up to {maxPriceChange}')








