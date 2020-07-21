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

from logging import getLogger
import time
import statsmodels.api as sm
import pandas as pd
import numpy as np
import scipy.optimize, scipy.stats
from tqdm import tqdm
import pickle
import datetime

from . import price_income
from .clear_market import clear_market
from eqsormo.common import BaseSortingModel, MNLFullASC
from eqsormo.common.util import human_bytes, human_time, human_shape
from eqsormo.common.compute_ascs import compute_ascs
import eqsormo

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

    def __init__ (self, housing_attributes, household_attributes,  interactions, unequilibrated_hh_params, unequilibrated_hsg_params, second_stage_params, price, income, choice, unequilibrated_choice, price_income_transformation=price_income.logdiff,
            price_income_starting_values=[], sample_alternatives=None, method='L-BFGS-B', max_rent_to_income=None, household_housing_attributes=None, weights=None, max_chunk_bytes=2e9):
        '''
        Initialize a Tra sorting model

        :param housing_attributes: Attributes of housing choices. Price should not be an attribute here.
        :type housing_attributes: Pandas dataframe.

        :param household_attributes: Attributes of households. It is okay if income is an attribute here as long as it is also passed in separately - since we assume it doesn't change when a household moves.
        :type household_attributes: Pandas dataframe

        :param household_housing_attributes: Attributes of the combination of households and housing choices, multiindexed with household ID and choice ID. Must contain all possible combinations.
        :type household_housing_attributes: Pandas dataframe

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

        :param max_chunk_bytes: Maximum number of bytes to use for a single chunk of the full alternatives array. The most memory-constrained part of the alogorithm
        is clearing the market, which requires computing the dot product of a n_households x n_housing_alternatives * n_unequilibrated_alternatives array and a coefficients
        array. However, since the coefficients array is a column vector, we can compute the product of chunks of the full alternatives array and the column vector, then concatenate.
        This tuning parameter controls how large these chunks are in bytes. Default 2GB.
        :type max_chunk_bytes: int
        '''
        self.housing_attributes = housing_attributes
        self.household_attributes = household_attributes
        self.household_housing_attributes = household_housing_attributes
        self.interactions = interactions
        self.second_stage_params = second_stage_params
        self.price = price.reindex(housing_attributes.index)
        self.orig_price = self.price.copy()
        self.income = income.reindex(household_attributes.index)
        self.choice = choice.reindex(household_attributes.index)
        self.unequilibrated_choice = unequilibrated_choice.reindex(household_attributes.index)
        self.unequilibrated_hh_params = unequilibrated_hh_params
        self.unequilibrated_hsg_params = unequilibrated_hsg_params
        self.sample_alternatives = sample_alternatives
        self.alternatives_stds = None
        
        if weights is None:
            self.weights = None
        else:
            self.weights = weights.reindex(household_attributes.index)

        self.price_income_transformation = price_income_transformation
        self.price_income_starting_values = price_income_starting_values
        self.method = method
        self.max_rent_to_income = max_rent_to_income
        self.max_chunk_bytes = max_chunk_bytes

        self._rng = np.random.default_rng()

        self.validate()

        self.creation_time = datetime.datetime.today()

    def validate (self):
        allPassed = True

        choiceCount = self.choice.value_counts().reindex(self.housing_attributes.index, fill_value=0)
        if not np.all(choiceCount > 0):
            choiceList = ' - ' + '\n - '.join(choiceCount.index[choiceCount == 0])
            LOG.error(f'Some housing alternatives are not chosen by any households!\n{choiceList}')
            allPassed = False
        
        if self.max_rent_to_income is not None and not np.all(self.income.values * self.max_rent_to_income > self.price.loc[self.choice].values):
            LOG.error('Some households pay more in rent than the max rent to income ratio')
            allPassed = False

        for hhattr, hsgattr in self.interactions:
            if hhattr not in self.household_attributes.columns:
                LOG.error(f'Attribute {hhattr} is used in interactions but is not in household_attributes')
                allPassed = False

            if self.household_attributes[hhattr].isnull().any():
                LOG.error(f'Attribute {hhattr} contains NaNs')
                allPassed = False

            if hsgattr not in self.housing_attributes.columns:
                LOG.error(f'Attribute {hsgattr} is used in interactions but is not in housing_attributes')
                allPassed = False

            if self.housing_attributes[hsgattr].isnull().any():
                LOG.error(f'Attribute {hsgattr} contains NaNs')
                allPassed = False

        if self.second_stage_params is not None:
            for hsgattr in self.second_stage_params:
                if hsgattr not in self.housing_attributes.columns:
                    LOG.error(f'Attribute {hsgattr} is used in second stage but is not in housing_attributes')
                    allPassed = False

                if self.housing_attributes[hsgattr].isnull().any():
                    LOG.error(f'Attribute {hsgattr} contains NaNs')
                    allPassed = False

        # TODO more checks
        if allPassed:
            LOG.info('All validation checks passed!')
        else:
            raise ValueError('Some validation checks failed (see log messages)')

    def materialize_alternatives (self, hhidx, choiceidx, uneqchoiceidx):
        '''
        Materialize the alternatives for hhidx, choiceidx, and uneqchoiceidx, and return them.

        These should be formatted like so, with hhidx changing slowest and uneqchoiceidx changing fastest.
        if there are three households, three housing choices, and three unequilibrated choices:
        hhidx:         0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2
        choiceidx:     0 0 0 1 1 1 2 2 2 0 0 0 1 1 1 2 2 2 0 0 0 1 1 1 2 2 2
        uneqchoiceidx: 0 1 2 0 1 2 0 1 2 0 1 2 0 1 2 0 1 2 0 1 2 0 1 2 0 1 2

        It is okay if they are not sequential, but they should be monotonically increasing.
        '''
        start_time = time.perf_counter()

        assert len(hhidx) == len(choiceidx) and len(choiceidx) == len(uneqchoiceidx)

        LOG.info(f'materializing {len(hhidx)} choices')

        # first, create data for the interactions
        colnames = []

        # + 1 for budget param
        ncols = len(self.interactions) + (len(self.unequilibrated_hh_params) +\
            len(self.unequilibrated_hsg_params)) * (len(self.unequilibrated_choice_xwalk) - 1) +\
                 1
        LOG.info(f'Allocating full alternatives array of size {human_bytes(len(hhidx) * ncols * 8)}')
        alternatives = np.zeros((len(hhidx), ncols))

        # budget is first column, to make updates easier
        current_col = 0
        colnames.append('budget')
        alt_income = self.income.astype('float64').values[hhidx]
        alt_price = self.price.astype('float64').values[choiceidx]

        # don't calc buget for options not in choice set
        # it may throw an error (e.g. log(neg) for logdiff)
        if self.max_rent_to_income is not None:
            feasible_alts = alt_income * self.max_rent_to_income > alt_price
        else:
            feasible_alts = np.full(len(alt_income), True)
        
        budget = np.full(len(hhidx), np.nan)
        budget[feasible_alts] = self.price_income_transformation.apply(alt_income[feasible_alts], alt_price[feasible_alts], *self.price_income_starting_values)
        alternatives[:,current_col] = budget
        current_col += 1
        del alt_income, alt_price, budget # save memory

        for hh_attr, hsg_attr in self.interactions:
            alternatives[:,current_col] =\
                self.household_attributes[hh_attr].astype('float64').values[hhidx] * self.housing_attributes[hsg_attr].astype('float64').values[choiceidx]
            colnames.append(f'{hh_attr}:{hsg_attr}')
            current_col += 1
        
        # now add the attributes for the unequilibrated choice
        for param in self.unequilibrated_hh_params:
            vals = self.household_attributes[param].astype('float64').values[hhidx]
            for uneqchoice in range(1, len(self.unequilibrated_choice_xwalk)):
                # fill all rows that are not for this unequilibrated choice with 0s
                alternatives[:,current_col] = np.choose(uneqchoiceidx == uneqchoice, [0, vals])
                colnames.append(f'{param}:uneq_choice_{self.unequilibrated_choice_xwalk[self.unequilibrated_choice_xwalk == uneqchoice].index[0]}')
                current_col += 1

        for param in self.unequilibrated_hsg_params:
            vals = self.housing_attributes[param].astype('float64').values[choiceidx]
            for uneqchoice in range(1, len(self.unequilibrated_choice_xwalk)):
                # fill all rows that are not for this unequilibrated choice with 0s
                alternatives[:,current_col] = np.choose(uneqchoiceidx == uneqchoice, [0, vals])
                colnames.append(f'{param}:uneq_choice_{self.unequilibrated_choice_xwalk[self.unequilibrated_choice_xwalk == uneqchoice].index[0]}')
                current_col += 1

        total_time = time.perf_counter() - start_time
        LOG.info(f'Materialized alternatives into {human_shape(alternatives.shape)} array using {human_bytes(alternatives.nbytes)} in {human_time(total_time)}')
        self.alternatives_colnames = colnames # hacky to set this every time but it never changes

        if self.alternatives_stds is None:
            self.alternatives_stds = np.std(alternatives)
        
        alternatives /= self.alternatives_stds

        return alternatives

    def create_alternatives (self):
        LOG.info('Creating alternatives')
        startTime = time.perf_counter()

        LOG.info('Converting pandas data to numpy')
        self.housing_xwalk = pd.Series(np.arange(len(self.housing_attributes)), index=self.housing_attributes.index)

        # we always have an unequilibrated choice to simplify coding, it is just only a single choice if not specifice
        # good ol' mononomial logit model
        unequilibrated_choice = self.unequilibrated_choice.copy() if self.unequilibrated_choice is not None else pd.Series(np.zeros(len(choice), index=choice.index))
        unique_unequilibrated_choices = unequilibrated_choice.unique()
        self.unequilibrated_choice_xwalk = pd.Series(np.arange(len(unique_unequilibrated_choices)), index=unique_unequilibrated_choices)
        self.hh_xwalk = pd.Series(np.arange(len(self.household_attributes)), index=self.household_attributes.index)

        self.hh_hsg_choice = self.housing_xwalk.loc[self.choice.loc[self.hh_xwalk.index]].values
        self.hh_unequilibrated_choice = self.unequilibrated_choice_xwalk.loc[self.unequilibrated_choice.loc[self.hh_xwalk.index]].values

        # index of each household in the full alternatives dataset
        # repeated for each alternative (housing choice)
        # so if there are three households, three housing choices, and three unequilibrated choices:
        # hhidx:         0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2
        # choiceidx:     0 0 0 1 1 1 2 2 2 0 0 0 1 1 1 2 2 2 0 0 0 1 1 1 2 2 2
        # uneqchoiceidx: 0 1 2 0 1 2 0 1 2 0 1 2 0 1 2 0 1 2 0 1 2 0 1 2 0 1 2
        # NB this could also be conceptualized as a four-dimensional array (household * housing choice * unequilibrated choice * variables)
        # but, uh, let's not
        LOG.info('Indexing full alternatives dataset')
        self.full_hhidx = np.repeat(np.arange(len(self.household_attributes)), len(self.housing_attributes) * len(unique_unequilibrated_choices))
        self.full_choiceidx = np.repeat(np.tile(np.arange(len(self.housing_attributes)), len(self.household_attributes)), len(unique_unequilibrated_choices))
        self.full_uneqchoiceidx = np.tile(np.arange(len(unique_unequilibrated_choices)), len(self.housing_attributes) * len(self.household_attributes))
        self.full_hsgchosen = (self.hh_hsg_choice[self.full_hhidx] == self.full_choiceidx) 
        self.full_uneqchosen = (self.hh_unequilibrated_choice[self.full_hhidx] == self.full_uneqchoiceidx)
        self.full_chosen = self.full_hsgchosen & self.full_uneqchosen

        if self.sample_alternatives <= 0 or self.sample_alternatives is None:
            if self.max_rent_to_income is None:
                self.alternatives = self.materialize_alternatives(self.full_hhidx, self.full_choiceidx, self.full_uneqchoiceidx)
                self.alternatives_hhidx = self.full_hhidx
                self.alternatives_choiceidx = self.full_choiceidx
                self.alternatives_uneqchoiceidx = self.full_uneqchoiceidx
                self.alternatives_hsgchosen = self.full_hsgchosen
                self.alternatives_uneqchosen = self.full_uneqchosen
                self.alternatives_chosen = self.full_chosen
            else:
                # TODO
                raise ValueError('max_rent_to_income with no sampling is unimplemented')

        else:
            LOG.info('Sampling alternatives')
            if self.max_rent_to_income is None:
                # note that we do not include the other unequilibrated choises for the chosen housing unit here, so they are not selected
                # randomly. We are randomly sampling housing alternatives, but always use all unequilibrated alternatives.
                feasible_unchosen_alts = ~self.full_hsgchosen
            else:
                feasible_unchosen_alts = (self.income.astype('float64').values[self.full_hhidx] * self.max_rent_to_income >\
                    self.price.astype('float64').values[self.full_choiceidx]) & ~self.full_hsgchosen

            # unequilibrated alternatives are not sampled
            n_housing_alts_per_hh = np.bincount(self.full_hhidx[feasible_unchosen_alts]) / len(self.unequilibrated_choice_xwalk)

            def random_sel (n):
                if n <= self.sample_alternatives - 1:
                    return np.repeat([True], n)
                else:
                    ret = np.arange(n) < self.sample_alternatives - 1
                    self._rng.shuffle(ret)
                    return ret

            # since households are the outermost index, and unequilibrated choices are the innermost index, we can get away with this
            # the repeat() makes sure all unquilibrated alternatives are selected (recall they are always adjacent), and concatenating
            # is correct since households are the slowest-changing index
            sampled_mask = np.concatenate([
                np.repeat(random_sel(n), len(self.unequilibrated_choice_xwalk))
                for n in n_housing_alts_per_hh
            ])

            unchosen_sampled_idxs = np.arange(len(self.full_hhidx))[feasible_unchosen_alts][sampled_mask]
            del sampled_mask
            chosen_idxs = np.arange(len(self.full_hhidx))[self.full_hsgchosen] # we do not sample uneq alternatives

            sampled_idxs = np.concatenate([unchosen_sampled_idxs, chosen_idxs])
            # put them back in the household > housing > uneq order
            np.sort(sampled_idxs)

            self.alternatives_hhidx = self.full_hhidx[sampled_idxs]
            self.alternatives_choiceidx = self.full_choiceidx[sampled_idxs]
            self.alternatives_uneqchoiceidx = self.full_uneqchoiceidx[sampled_idxs]
            self.alternatives_hsgchosen = self.full_hsgchosen[sampled_idxs]
            self.alternatives_uneqchosen = self.full_uneqchosen[sampled_idxs]
            self.alternatives_chosen = self.full_chosen[sampled_idxs]
            self.alternatives = self.materialize_alternatives(self.alternatives_hhidx, self.alternatives_choiceidx, self.alternatives_uneqchoiceidx)

        endTime = time.perf_counter()
        LOG.info(f'Created alternatives for {len(self.household_attributes)} households in {endTime - startTime:.3f} seconds')
        LOG.info(f'Alternatives dimensions: {human_shape(self.alternatives.shape)}')
        LOG.info(f'Alternatives use {human_bytes(self.alternatives.nbytes)} memory')

    def first_stage_utility (self, params):
        if self.price_income_transformation.n_params > 0:
            coefs = params[:-self.price_income_transformation.n_params]
            transformation_params = params[-self.price_income_transformation.n_params:]
        else:
            coefs = params
            transformation_params = []

        if not np.all(np.isfinite(self.alternatives[0,:])):
            raise ValueError('not all budgets are finite')

        if len(transformation_params) > 0:
            # recalc budgets if they are dependent on an estimated parameter
            self.alternatives[0,:] = self.price_income_transformation.apply(
                self.income.astype('float64').values[self.alternatives_hhidx],
                self.price.astype('float64').values[self.alternatives_choiceidx],
                *transformation_params)

        # compute utilities
        utils = np.dot(self.alternatives, coefs).reshape(-1)

        # add log of supply, which is the part of the ASC which reacts to changing market shares
        # unequilibrated alternatives do not
        # TODO could move this calculation into MNLFullASC
        utils += self._log_supply
        
        return utils
        
    def fit_first_stage (self):
        LOG.info('fitting first stage')

        self._log_supply = np.log(np.bincount(self.hh_hsg_choice))[self.alternatives_choiceidx]

        self.first_stage_fit = MNLFullASC(
            utility=self.first_stage_utility,
            choiceidx=(self.alternatives_choiceidx, self.alternatives_uneqchoiceidx),
            hhidx=self.alternatives_hhidx,
            chosen=self.alternatives_chosen,
            supply=(
                np.bincount(self.hh_hsg_choice),
                np.bincount(self.hh_unequilibrated_choice)
            ),
            starting_values=np.concatenate([np.zeros(self.alternatives.shape[1]), self.price_income_transformation.starting_values]),
            param_names=[*self.alternatives_colnames, *self.price_income_transformation.param_names],
            method=self.method
        )

        self.first_stage_fit.fit()

        # recalculate ASCs to clear full market
        # to make the model tractable, we estimate on a sample of the alternatives - which provides consistent but inefficient parameter estimates
        # (see Ben-Akiva and Lerman 1985)
        # Since when we clear the market in scenario evaluation, we use the full set of alternatives, not just the sampled alternatives, we want
        # the baseline (current conditions) to be market clearing as well, we re-estimate the ASCs with the full set of alternatives
        # another way to look at this is that due to tractability concerns we cannot estimate the full model without alternative sampling,
        # so we lose some efficiency - but we can estimate the ASCs without alternative sampling, so we don't lose the efficiency there.

        if self.max_rent_to_income is None:
            feasible_alts = np.full(True, len(self.full_hhidx))
        else:
            feasible_alts = (self.income.astype('float64').values[self.full_hhidx] * self.max_rent_to_income >\
                self.price.astype('float64').values[self.full_choiceidx])

        coefs = self.first_stage_fit.params.values[:-self.price_income_transformation.n_params]

        # compute utility in blocks to save memory. we can do this because we don't actually need to materialize the full_alternatives matrix
        # to cumpute the utility - and in the base case, the full alternatives matrix is 67 GB. Instead, we materialize chunks, compute the dot
        # product with the coefs for those alternatives, and iteratively update
        base_utility = np.full(len(self.full_hhidx), np.nan)
        chunk_rows = int(np.floor(self.max_chunk_bytes / len(self.alternative_colnames) / 8)) # bytes per float64

        LOG.info(f'Computing full utilities using {len(self.full_hhidx) // chunk_rows + 1} chunks of {chunk_rows} rows each ({human_bytes(chunk_rows * len(self.alternative_colnames) * 8)} each)')

        fullAscStartTime = time.perf_counter()

        # compute these with weights. This should be okay because the unweighted estimates are consistent if the sampling is conditional
        # on the choices, and with all the other assumptions we're making we might as well make that one as well... we're not doing much with
        # the second stage estimates anyhow
        weighted_supply = np.bincount(self.hh_hsg_choice, self.weights.values)
        for chunk_start in range(0, len(self.full_hhidx), chunk_rows):
            chunk_end = min(chunk_start + chunk_rows, len(self.full_hhidx))
            chunk_alts = self.materialize_alternatives(
                self.full_hhidx[chunk_start:chunk_end],
                self.full_choiceidx[chunk_start:chunk_end],
                self.full_uneqchoiceidx[chunk_start:chunk_end]
            )
            # add systematic utility and deterministic part of ASC based on market share
            base_utility[chunk_start:chunk_end] = np.dot(chunk_alts, coefs) +\
                np.log(weighted_supply)[self.full_choiceidx[chunk_start:chunk_end]]

        ascs = compute_ascs(
            base_utilities=base_utility[feasible_alts],
            supply=(
                # of homes
                weighted_supply,
                # of unequilibrated choices
                # While we don't (obviously) equilibrate unequilibrated choices, we do use their supply to find their ASCs in the first-stage fit
                np.bincount(self.hh_unequilibrated_choice, self.weights.values)
            ),
            hhidx=self.full_hhidx[feasible_alts],
            choiceidx=(
                self.full_choiceidx[feasible_alts],
                self.full_uneqchoiceidx[feasible_alts]
            ),
            starting_values=self.first_stage_fit.ascs,
            weights=self.weights.values
        )
        fullAscEndTime = time.perf_counter()

        # descale coefs
        # but don't descale transformation parameters
        # recall that the _result_ of the transformation, not the inputs, is what is scaled, so this is okay
        scalars = pd.Series(self.alternatives_stds, index=self.alternatives_colnames)
        scalars.loc[self.price_income_transformation.param_names] = 1
        scalars = scalars.reindex(self.first_stage_fit.params.index)
        self.first_stage_fit.params /= scalars
        self.first_stage_fit.se /= scalars

        # TODO make pd.series
        self.first_stage_ascs = ascs
        LOG.info(f'Finding full ASCs took {fullAscEndTime - fullAscStartTime:.3f}s')

    def fit_second_stage (self):
        LOG.info('fitting second stage')
        startTime = time.perf_counter()
        second_stage_exog = sm.add_constant(self.housing_attributes[self.second_stage_params])
        second_stage_endog = self.first_stage_ascs.reindex(second_stage_exog.index)

        mod = sm.OLS(second_stage_endog, second_stage_exog)
        self.second_stage_fit = mod.fit()
        self.type_shock = self.second_stage_fit.resid
        endTime = time.perf_counter()
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
        if self.household_housing_attributes is not None:
            otherInteractions = {vname: self.fullAlternatives[vname] for vname in self.household_housing_attributes}
        else:
            otherInteractions = dict()

        full_first_stage_data = pd.DataFrame({**{
            # demeaning b/c it was done in the Bayer paper - and since we don't sample from households no concerns about using the mean
            f'{household}:{housing}': (self.fullAlternatives[household] - self.household_attributes[household].mean()) * self.fullAlternatives[housing]
            for household, housing in self.interactions
        }, **otherInteractions})

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
        non_price_utilities = base_utility +\
            self.first_stage_ascs.loc[choice_xwalk.index].values[choiceidx] +\
            np.log(self.weighted_supply.loc[choice_xwalk.index].values[choiceidx])

        startTimeClear = time.perf_counter()
        new_prices = clear_market(
            non_price_utilities=non_price_utilities,
            hhidx=hhidx,
            choiceidx=choiceidx,
            supply=self.weighted_supply.loc[choice_xwalk.index].values, # weighted supply is identical to unweighted supply when there are no weights
            income=self.income.loc[hh_xwalk.index].values,
            starting_price=self.price.loc[choice_xwalk.index].values,
            price_income_transformation=self.price_income_transformation,
            price_income_params=transformationParams,
            budget_coef=self.first_stage_fit.params['budget'],
            max_rent_to_income=self.max_rent_to_income,
            maxiter=maxiter,
            weights=self.weights.loc[hh_xwalk.index].values if self.weights is not None else None
        )
        endTimeClear = time.perf_counter()

        new_prices = pd.Series(new_prices, index=choice_xwalk.index)

        maxPriceChange = np.max(np.abs(new_prices - self.price))

        LOG.info(f'Market cleared in {endTimeClear - startTimeClear:.2f}s, with price changes up to {maxPriceChange:.4f}')

        self.price = new_prices

    def probabilities (self):
        if self.household_housing_attributes is not None:
            otherInteractions = {vname: self.fullAlternatives[vname] for vname in self.household_housing_attributes}
        else:
            otherInteractions = dict()
        
        full_first_stage_data = pd.DataFrame({**{
            # demeaning b/c it was done in the Bayer paper - and since we don't sample from households no concerns about using the mean
            f'{household}:{housing}': (self.fullAlternatives[household] - self.household_attributes[household].mean()) * self.fullAlternatives[housing]
            for household, housing in self.interactions
        }, **otherInteractions})

        if self.max_rent_to_income is not None:
            full_first_stage_data = full_first_stage_data[
                self.fullAlternatives.income.reindex(full_first_stage_data.index) * self.max_rent_to_income >\
                    self.price.reindex(full_first_stage_data.index, level='choice')]

        if self.price_income_transformation.n_params > 0:
            coefs = self.first_stage_fit.params.iloc[:-self.price_income_transformation.n_params]
            transformationParams = self.first_stage_fit.params.values[-self.price_income_transformation.n_params:]
        else:
            coefs = self.first_stage_fit.params
            transformationParams = np.array([])

        full_first_stage_data['budget'] = self.price_income_transformation.apply(
            self.fullAlternatives.income.loc[full_first_stage_data.index],
            self.price.reindex(full_first_stage_data.index, level='choice'),
            *transformationParams)

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

        # NB not multiplying by weights here. This is the choice probability, not the market share

        return pd.Series(probs, index=full_first_stage_data.index).rename('choice_prob')

    def to_text (self, fn=None):
        "Save model results as text. If fn==None, return as string"

        outstring = '''Equilibrium sorting model (Tra [2007] formulation, price in first stage as budget constraint)
Model run initiated at {creation_time}
Budget function {price_income_transformation}

First stage (discrete choice sorting model):
{first_stage_summary}

Second stage (OLS parameters):
{second_stage_summary}

Fit with EqSorMo version {version}, https://github.com/mattwigway/eqsormo            
        '''.format(
            first_stage_summary=self.first_stage_fit.summary(),
            second_stage_summary=self.second_stage_fit.summary(),
            creation_time=self.creation_time.strftime('%Y-%m-%d %H:%M:%S %Z'),
            price_income_transformation=self.price_income_transformation.name,
            version=eqsormo.version
        )

        if fn is not None:
            with open(fn, 'w') as outfile:
                outfile.write(outstring)
        else:
            return outstring

    # don't pickle fullAlternatives
    def __getstate__ (self):
        return {k: v for k, v in self.__dict__.items() if k != 'fullAlternatives'}

    @classmethod
    def from_pickle(cls, fn):
        tra = super().from_pickle(fn)
        tra.create_full_alternatives()
        return tra


    




