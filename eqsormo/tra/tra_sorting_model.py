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
from eqsormo.common import BaseSortingModel, MNLFullASC

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

    def __init__ (self, housing_attributes, household_attributes, interactions, second_stage_params, price, income, choice, price_income_transformation=price_income.logdiff, sample_alternatives=None, method='bfgs'):
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
        self.validate()

    def validate (self):
        allPassed = True

        choiceCount = self.choice.value_counts().reindex(self.housing_attributes.index, fill_value=0)
        if not np.all(choiceCount > 0):
            choiceList = ' - ' + '\n - '.join(choiceCount.index[choiceCount == 0])
            LOG.error(f'Some housing alternatives are not chosen by any households!\n{choiceList}')
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

        self.fullAlternatives = self.fullAlternatives[self.fullAlternatives.income > self.fullAlternatives.price].copy()

        LOG.info('created full set of alternatives, now sampling if requested')

        if self.sample_alternatives <= 0 or self.sample_alternatives is None:
            self.alternatives = self.fullAlternatives
        else:
            unchosenAlternatives = self.fullAlternatives[~self.fullAlternatives.chosen].groupby(level=0).apply(lambda x: x.sample(self.sample_alternatives - 1) if len(x) >= self.sample_alternatives else x)
            unchosenAlternatives.index = unchosenAlternatives.index.droplevel(0) # fix dup'd household level due to groupby
            self.alternatives = pd.concat([unchosenAlternatives, self.fullAlternatives[self.fullAlternatives.chosen]]).sort_index(level=[0, 1])

        self.alternatives.drop(columns=['chosen', 'hhchoice'], inplace=True)
        self.fullAlternatives.drop(columns=['chosen', 'hhchoice'], inplace=True)

        self.alternatives.index.rename(['household', 'choice'], inplace=True)

        endTime = time.clock()
        LOG.info(f'Created {len(self.alternatives)} alternatives for {len(self.household_attributes)} households in {endTime - startTime:.3f} seconds')

    def first_stage_utility (self, params):
        # todo is this the right self?
        if self.price_income_transformation.n_params > 0:
            coefs = params[:-self.price_income_transformation.n_params]
            transformationParams = params[-self.price_income_transformation.n_params:]
        else:
            coefs = params
            transformationParams = []

        # don't recalc sd each time, so that the scale of the budget variable is constant in transformation space
        self._first_stage_data['budget'] = self.price_income_transformation\
            .apply(self.alternatives.income, self.alternatives.price, *transformationParams) / self._first_stage_stdevs['budget']

        # compute utilities
        #namedCoefs = pd.Series(coefs, index=self._first_stage_data.columns)
        return self._first_stage_data.multiply(coefs, axis='columns').sum(axis='columns')
        
    def fit_first_stage (self):
        LOG.info('fitting first stage')
        
        # create the data for the first stage
        self._first_stage_data = pd.DataFrame({
            # demeaning b/c it was done in the Bayer paper
            f'{household}:{housing}': (self.alternatives[household] - self.household_attributes[household].mean()) * self.alternatives[housing]
            for household, housing in self.interactions
        })

        self._first_stage_data['budget'] = self.price_income_transformation\
            .apply(self.alternatives.income, self.alternatives.price, *self.price_income_transformation.starting_values)

        # Note that budget will be rescaled by this standard deviation, not its new standard deviation when updated, so that the model does not get confused trying to
        # chase a variable whose scale is changing. This is just a hack to make the model converge, so it's not important that everything have standard deviation of exactly 1.
        self._first_stage_stdevs = self._first_stage_data.apply(np.std)
        self._first_stage_data /= self._first_stage_stdevs

        self.first_stage_fit = MNLFullASC(
            utility=self.first_stage_utility,
            choice=self.choice,
            supply=self.supply,
            starting_values=np.zeros(len(self._first_stage_data.columns) + self.price_income_transformation.n_params),
            param_names=[*self._first_stage_data.columns, *self.price_income_transformation.param_names],
            method=self.method
        )

        self.first_stage_fit.fit()

    def fit_second_stage (self):
        LOG.info('fitting second stage')
        startTime = time.clock()
        self._second_stage_exog = sm.add_constant(self.housing_attributes[self.second_stage_params])
        self._second_stage_endog = self.first_stage_fit.ascs.reindex(self._second_stage_exog.index)

        mod = sm.OLS(self._second_stage_endog, self._second_stage_exog)
        self.second_stage_fit = mod.fit()
        endTime = time.clock()
        LOG.info(f'Fit second stage in {endTime - startTime:.2f} seconds')

    def fit (self):
        self.fit_first_stage()
        if self.second_stage_params is not None:
            self.fit_second_stage()
        else:
            LOG.info('No second stage requested')




