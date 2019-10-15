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
import pickle

from . import price_income
from .clear_market import clear_market
from eqsormo.common import BaseSortingModel, MNLFullASC
from eqsormo.common.compute_ascs import compute_ascs

LOG = getLogger(__name__)

class TraSortingModel(BaseSortingModel):
    '''
    The variety of sorting model described in Tra CI (2013) Measuring the General Equilibrium Benefits of Air Quality Regulation in Small Urban Areas. Land Economics 89(2): 291–307.

    The key difference between this and the Bayer et al. model is that price is included in the first stage rather than the second stage. This obviates the need to instrument
    for price, because there are housing type fixed effects in the first stage. But we can't include price directly, due to identification
    challenges since the prices could also be captured by the ASCs. So instead we include a budget constraint of the form f(income, price) - Tra uses
    ln(income - price) but the exact functional form is not important, as long as income (or another household attribute) is somehow interacted with price
    so that there is variation in price between households 
    '''

    def __init__ (self, housing_attributes, household_attributes, interactions, second_stage_params, price, income, choice, price_income_transformation=price_income.logdiff,
            sample_alternatives=None, method='L-BFGS-B', max_rent_to_income=None):
        '''
        Initialize a Tra sorting model

        :param housing_attributes: Attributes of housing choices. Price should not be an attribute here.
        :type housing_attributes: Pandas dataframe.

        :param household_attributes: Attributes of households. It is okay if income is an attribute here as long as it is also passed in separately - since we assume it doesn't change when a household moves.
        :type household_attributes: Pandas dataframe

        :param interactions: Which household attributes should be interacted with which housing attributes in the first stage
        :type interactions: iterable of tuple (household_attributes column, housing_attributes column)

        :params second_stage_params: Which housing attributes to include in the second stage fit (decomposition of mean parameters). If None, don't fit a second stage (this is not necessary for simulation).
        :type second_stage_params: iterable of string

        :param price: Observed price of each housing choice. This price should be over the same time period as income (for example, annualized prices with annual incomes)
        :type price: Pandas series of numbers, indexed like housing_attributes

        :param income: Household income
        :type income: Pandas series of numbers, indexed like household_attributes
        
        :param choice: The choice made by the household. Should be index values in household_attributes
        :param type: Pandas series, indexed like household_attributes

        :param sample_alternatives: If a positive integer, take a random sample of this size for available alternatives to estimate the model. Recommended in models with a large number of alternatives. If 0 or None, no sampling will be performed. Use np.set_seed() before calling fit for model reproducibility when using this option.
        :type sample_alternatives: int or None

        :param method: A scipy.optimize.minimize method used for maximizing log-likelihood of first stage, default 'bfgs'

        :param max_rent_to_income: Maximum proportion of income rent can be for the alternative to be included in the choice set - in (0, 1] if the price/income transformation can't handle income less than rent
        :type max_rent_to_income: float or None
        '''
        self.housing_attributes = housing_attributes.copy()
        self.household_attributes = household_attributes.copy()
        self.interactions = interactions
        self.second_stage_params = second_stage_params
        self.price = price.copy()
        self.orig_price = price.copy()
        self.income = income.copy()
        self.choice = choice.copy()
        self.sample_alternatives = sample_alternatives
        self.supply = choice.value_counts()
        self.price_income_transformation = price_income_transformation
        self.method = method
        self.max_rent_to_income = max_rent_to_income
        self.validate()

    def validate (self):
        allPassed = True

        choiceCount = self.choice.value_counts().reindex(self.housing_attributes.index, fill_value=0)
        if not np.all(choiceCount > 0):
            choiceList = ' - ' + '\n - '.join(choiceCount.index[choiceCount == 0])
            LOG.error(f'Some housing alternatives are not chosen by any households!\n{choiceList}')
            allPassed = False
        
        if self.max_rent_to_income is not None and not np.all(self.income.values * self.max_rent_to_income > self.price.loc[self.choice.reindex(self.income.index)].values):
            LOG.error('Some households pay more in rent than the max rent to income ratio')
            allPassed = False

        # TODO more checks
        if allPassed:
            LOG.info('All validation checks passed!')

    def create_alternatives (self):
        LOG.info('Creating alternatives')
        startTime = time.clock()

        self.fullAlternatives = pd.concat([self.housing_attributes for i in range(len(self.household_attributes))], keys=self.household_attributes.index)
        self.fullAlternatives.index.rename(['household', 'choice'], inplace=True)
        self.household_attributes.index.rename('household', inplace=True)
        self.fullAlternatives = self.fullAlternatives.join(self.household_attributes, on='household')
        self.fullAlternatives = self.fullAlternatives.join(pd.DataFrame(self.price.rename('price')), on='choice')
        self.fullAlternatives = self.fullAlternatives.join(pd.DataFrame(self.income.rename('income')), on='household')
        self.fullAlternatives['chosen'] = False
        self.fullAlternatives['hhchoice'] = self.choice.reindex(self.fullAlternatives.index, level=0)
        self.fullAlternatives.loc[self.fullAlternatives.index.get_level_values(1) == self.fullAlternatives.hhchoice, 'chosen'] = True

        LOG.info('created full set of alternatives, now sampling if requested')

        if self.sample_alternatives <= 0 or self.sample_alternatives is None:
            if self.max_rent_to_income is None:
                self.alternatives = self.fullAlternatives
            else:
                self.alternatives = self.fullAlternatives[self.fullAlternatives.income * self.max_rent_to_income > self.fullAlternatives.price]

        else:
            if self.max_rent_to_income is None:
                candidateAlternatives = self.fullAlternatives
            else:
                candidateAlternatives = self.fullAlternatives[self.fullAlternatives.income * self.max_rent_to_income > self.fullAlternatives.price]

            unchosenAlternatives = candidateAlternatives[~candidateAlternatives.chosen].groupby(level=0).apply(lambda x: x.sample(self.sample_alternatives - 1) if len(x) >= self.sample_alternatives else x)
            unchosenAlternatives.index = unchosenAlternatives.index.droplevel(0) # fix dup'd household level due to groupby
            self.alternatives = pd.concat([unchosenAlternatives, self.fullAlternatives[self.fullAlternatives.chosen]]).sort_index(level=[0, 1])

        self.alternatives.drop(columns=['chosen', 'hhchoice'], inplace=True)
        self.fullAlternatives.drop(columns=['chosen', 'hhchoice'], inplace=True)

        self.alternatives.index.rename(['household', 'choice'], inplace=True)

        endTime = time.clock()
        LOG.info(f'Created {len(self.alternatives)} alternatives for {len(self.household_attributes)} households in {endTime - startTime:.3f} seconds')

    def first_stage_utility (self, params):
        if self.price_income_transformation.n_params > 0:
            coefs = params[:-self.price_income_transformation.n_params]
            transformationParams = params[-self.price_income_transformation.n_params:]
        else:
            coefs = params
            transformationParams = []

        # don't recalc sd each time, so that the scale of the budget variable is constant in transformation space
        self._first_stage_data['budget'] = self.price_income_transformation\
            .apply(self.alternatives.income.values, self.alternatives.price.values, *transformationParams) / self._first_stage_stdevs['budget']

        # compute utilities
        return np.dot(self._first_stage_data.values, coefs).reshape(-1)
        
    def fit_first_stage (self):
        LOG.info('fitting first stage')
        
        # create the data for the first stage
        self._first_stage_data = pd.DataFrame({
            # demeaning b/c it was done in the Bayer paper
            f'{household}:{housing}': (self.alternatives[household] - self.household_attributes[household].mean()) * self.alternatives[housing]
            for household, housing in self.interactions
        })

        self._first_stage_data['budget'] = self.price_income_transformation\
            .apply(self.alternatives.income.values, self.alternatives.price.values, *self.price_income_transformation.starting_values)

        # Note that budget will be rescaled by this standard deviation, not its new standard deviation when updated, so that the model does not get confused trying to
        # chase a variable whose scale is changing. This is just a hack to make the model converge, so it's not important that everything have standard deviation of exactly 1.
        self._first_stage_stdevs = self._first_stage_data.apply(np.std)
        self._first_stage_data /= self._first_stage_stdevs

        # reindex everything into numpy arrays
        choice_xwalk = pd.Series(np.arange(len(self.housing_attributes)), index=self.housing_attributes.index)
        hh_xwalk = pd.Series(np.arange(len(self.household_attributes)), index=self.household_attributes.index)

        choiceidx = choice_xwalk.loc[self._first_stage_data.index.get_level_values('choice')].values
        hhidx = hh_xwalk.loc[self._first_stage_data.index.get_level_values('household')].values
        chosen = np.array([self._first_stage_data.index.get_loc((hh, choice)) for hh, choice in zip(self.choice.index, self.choice)])

        self.first_stage_fit = MNLFullASC(
            utility=self.first_stage_utility,
            choiceidx=choiceidx,
            hhidx=hhidx,
            chosen=chosen,
            supply=self.supply.loc[choice_xwalk.index].values,
            starting_values=np.concatenate([np.zeros(len(self._first_stage_data.columns)), self.price_income_transformation.starting_values]),
            param_names=[*self._first_stage_data.columns, *self.price_income_transformation.param_names],
            method=self.method
        )

        self.first_stage_fit.fit()

        # descale coefs
        # but don't descale transformation parameters
        # recall that the _result_ of the transformation, not the inputs, is what is scaled, so this is okay
        self.first_stage_fit.params /= self._first_stage_stdevs.reindex(self.first_stage_fit.params.index, fill_value=1)
        self.first_stage_fit.se /= self._first_stage_stdevs.reindex(self.first_stage_fit.params.index, fill_value=1)

        # recalculate ASCs to clear full market
        # to make the model tractable, we estimate on a sample of the alternatives - which provides consistent but inefficient parameter estimates
        # (see Ben-Akiva and Lerman 1985)
        # Since when we clear the market in scenario evaluation, we use the full set of alternatives, not just the sampled alternatives, we want
        # the baseline (current conditions) to be market clearing as well, we re-estimate the ASCs with the full set of alternatives
        # another way to look at this is that due to tractability concerns we cannot estimate the full model without alternative sampling,
        # so we lose some efficiency - but we can estimate the ASCs without alternative sampling, so we don't lose the efficiency there.
        full_first_stage_data = pd.DataFrame({
            # demeaning b/c it was done in the Bayer paper - and since we don't sample from households no concerns about using the mean
            f'{household}:{housing}': (self.fullAlternatives[household] - self.household_attributes[household].mean()) * self.fullAlternatives[housing]
            for household, housing in self.interactions
        })

        if self.max_rent_to_income is not None:
            full_first_stage_data = full_first_stage_data[self.fullAlternatives.income.reindex(full_first_stage_data.index) * self.max_rent_to_income > self.price.reindex(full_first_stage_data.index, level='choice')]

        if self.price_income_transformation.n_params > 0:
            coefs = self.first_stage_fit.params.iloc[:-self.price_income_transformation.n_params]
            transformationParams = self.first_stage_fit.params.values[-self.price_income_transformation.n_params:]
        else:
            coefs = self.first_stage_fit.params
            transformationParams = np.array([])

        full_first_stage_data['budget'] = self.price_income_transformation\
            .apply(self.fullAlternatives.income.loc[full_first_stage_data.index], self.fullAlternatives.price[full_first_stage_data.index], *transformationParams)

        # params have been destandardized above, so no need to standardize this data
        # standardization is needed in estimation so that when the algorithm moves the param a tiny
        # bit, the exp(utility) stays finite

        base_utility = np.dot(full_first_stage_data[coefs.index].values, coefs.values)

        fullAscStartTime = time.clock()
        ascs = compute_ascs(
            base_utility,
            self.supply.loc[choice_xwalk.index].values,
            hh_xwalk.loc[full_first_stage_data.index.get_level_values('household')].values,
            choice_xwalk.loc[full_first_stage_data.index.get_level_values('choice')].values,
            starting_values=self.first_stage_fit.ascs
        )
        fullAscEndTime = time.clock()

        self.first_stage_ascs = pd.Series(ascs, index=choice_xwalk.index)
        LOG.info(f'Finding full ASCs took {fullAscEndTime - fullAscStartTime:.3f}s')

    def fit_second_stage (self):
        LOG.info('fitting second stage')
        startTime = time.clock()
        second_stage_exog = sm.add_constant(self.housing_attributes[self.second_stage_params])
        second_stage_endog = self.first_stage_ascs.reindex(second_stage_exog.index)

        mod = sm.OLS(second_stage_endog, second_stage_exog)
        self.second_stage_fit = mod.fit()
        self.type_shock = self.second_stage_fit.resid
        endTime = time.clock()
        LOG.info(f'Fit second stage in {endTime - startTime:.2f} seconds')

    def fit (self):
        self.fit_first_stage()
        if self.second_stage_params is not None:
            self.fit_second_stage()
        else:
            LOG.info('No second stage requested')

    def sort (self, maxiter=np.inf):
        'Clear the market after changes have been made to the data'
        LOG.info('Clearing the market and sorting households')
        LOG.info("There's nothing hidden in your head / the Sorting Hat can't see / so try me on and I will tell you / where you ought to be.\n" +\
            "    -JK Rowling, Harry Potter and the Sorcerer's Stone")

        # first update second stage
        if self.second_stage_params is not None:
            LOG.info('updating second stage')
            pred_ascs = self.second_stage_fit.predict(sm.add_constant(self.housing_attributes[self.second_stage_params])) + self.type_shock
            
            maxabsdiff = np.max(np.abs(pred_ascs - self.first_stage_ascs))
            LOG.info(f'Second stage updated with changes to first-stage ASCs of up to {maxabsdiff:.2f}')
            self.first_stage_ascs = pred_ascs
        else:
            LOG.info('No second stage fit, not updating')
            pred_ascs = self.first_stage_ascs

        # then update the first stage
        full_first_stage_data = pd.DataFrame({
            # demeaning b/c it was done in the Bayer paper - and since we don't sample from households no concerns about using the mean
            f'{household}:{housing}': (self.fullAlternatives[household] - self.household_attributes[household].mean()) * self.fullAlternatives[housing]
            for household, housing in self.interactions
        })

        # Note that I am intentionally _not_ dropping hh/choice combinations here that do not meet rent to income criteria, because which households those are
        # might change in the sorting phase. The filtering happens there. This does not matter for the calculation of utility below, since utilities for
        # different alternatives are independent of each other and the budget is set to zero anyhow.
        if self.price_income_transformation.n_params > 0:
            coefs = self.first_stage_fit.params.iloc[:-self.price_income_transformation.n_params]
            transformationParams = self.first_stage_fit.params.values[-self.price_income_transformation.n_params:]
        else:
            coefs = self.first_stage_fit.params
            transformationParams = np.array([])

        # set the budget utility to zero for all households - and solve for below
        full_first_stage_data['budget'] = 0

        choice_xwalk = pd.Series(np.arange(len(self.housing_attributes)), index=self.housing_attributes.index)
        hh_xwalk = pd.Series(np.arange(len(self.household_attributes)), index=self.household_attributes.index)

        choiceidx = choice_xwalk.loc[full_first_stage_data.index.get_level_values('choice')].values
        hhidx = hh_xwalk.loc[full_first_stage_data.index.get_level_values('household')].values

        base_utility = np.dot(full_first_stage_data[coefs.index].values, coefs.values)
        non_price_utilities = base_utility + self.first_stage_ascs.loc[choice_xwalk.index].values[choiceidx]

        startTimeClear = time.clock()
        new_prices = clear_market(
            non_price_utilities=non_price_utilities,
            hhidx=hhidx,
            choiceidx=choiceidx,
            supply=self.supply.loc[choice_xwalk.index].values,
            income=self.income.loc[hh_xwalk.index].values,
            starting_price=self.price.loc[choice_xwalk.index].values,
            price_income_transformation=self.price_income_transformation,
            price_income_params=transformationParams,
            budget_coef=self.first_stage_fit.params['budget'],
            max_rent_to_income=self.max_rent_to_income,
            maxiter=maxiter
        )
        endTimeClear = time.clock()

        new_prices = pd.Series(new_prices, index=choice_xwalk.index)

        maxPriceChange = np.max(np.abs(new_prices - self.price))

        LOG.info(f'Market cleared in {endTimeClear - startTimeClear:.2f}s, with price changes up to {maxPriceChange:.4f}')

        self.price = new_prices

    def probabilities (self):
        full_first_stage_data = pd.DataFrame({
            # demeaning b/c it was done in the Bayer paper - and since we don't sample from households no concerns about using the mean
            f'{household}:{housing}': (self.fullAlternatives[household] - self.household_attributes[household].mean()) * self.fullAlternatives[housing]
            for household, housing in self.interactions
        })

        if self.max_rent_to_income is not None:
            full_first_stage_data = full_first_stage_data[self.fullAlternatives.income.reindex(full_first_stage_data.index) * self.max_rent_to_income > self.price.reindex(full_first_stage_data.index, level='choice')]

        if self.price_income_transformation.n_params > 0:
            coefs = self.first_stage_fit.params.iloc[:-self.price_income_transformation.n_params]
            transformationParams = self.first_stage_fit.params.values[-self.price_income_transformation.n_params:]
        else:
            coefs = self.first_stage_fit.params
            transformationParams = np.array([])

        full_first_stage_data['budget'] = self.price_income_transformation.apply(self.fullAlternatives.income.loc[full_first_stage_data.index], self.fullAlternatives.price.loc[full_first_stage_data.index], *transformationParams)

        choice_xwalk = pd.Series(np.arange(len(self.housing_attributes)), index=self.housing_attributes.index)
        hh_xwalk = pd.Series(np.arange(len(self.household_attributes)), index=self.household_attributes.index)

        choiceidx = choice_xwalk.loc[full_first_stage_data.index.get_level_values('choice')].values
        hhidx = hh_xwalk.loc[full_first_stage_data.index.get_level_values('household')].values

        utilities = np.dot(full_first_stage_data[coefs.index].values, coefs.values) + self.first_stage_ascs.loc[choice_xwalk.index].values[choiceidx]
        exp_utilities = np.exp(utilities)

        if not np.all(np.isfinite(exp_utilities)):
            raise ValueError('Non-finite utility detected!')

        logsums = np.bincount(hhidx, weights=exp_utilities)
        probs = exp_utilities / logsums[hhidx]

        return pd.Series(probs, index=full_first_stage_data.index).rename('choice_prob')



    




