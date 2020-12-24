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

# Author: Matthew Wigginton Conway <matt@indicatrix.org>
#         School of Geographical Sciences and Urban Planning
#         Arizona State University

from logging import getLogger
import time
import statsmodels.api as sm
import pandas as pd
import numpy as np
import datetime
import dask.array as da

from . import price_income
from .clear_market import clear_market_iter
from eqsormo.common import BaseSortingModel, MNLFullASC
from eqsormo.common.util import human_bytes, human_time, human_shape
from eqsormo.common.compute_ascs import compute_ascs
import eqsormo
from . import save_load

from typing import Dict, Callable, Any

LOG = getLogger(__name__)


class TraSortingModel(BaseSortingModel):
    """
    The variety of sorting model described in Tra CI (2013) Measuring the General Equilibrium Benefits of Air Quality
    Regulation in Small Urban Areas. Land Economics 89(2): 291–307.

    The key difference between this and the Bayer et al. model is that price is included in the first stage rather than
    the second stage. This obviates the need to instrument for price, because there are housing type fixed effects in
    the first stage. But we can't include price directly, due to identification challenges since the prices could also
    be captured by the ASCs. So instead we include a budget constraint of the form f(income, price) - Tra uses
    ln(income - price) but the exact functional form is not important, as long as income (or another household
    attribute) is somehow interacted with price so that there is variation in price between households within a single
    housing type.

    The other key difference from both the Bayer and the Tra formulations is that this model supports "unequilibrated
    choice," that is it can model a joint choice between a good for which prices are adjusted to bring the market to
    equilibrium and one for which they are not. In my dissertation I use this to model the joint choice of housing
    (equilibrated good) and vehicle ownership (unequilibrated good).
    """

    def __init__(
        self,
        housing_attributes,
        household_attributes,
        interactions,
        unequilibrated_hh_params,
        unequilibrated_hsg_params,
        second_stage_params,
        price,
        income,
        choice,
        unequilibrated_choice,
        price_income_transformation=price_income.logdiff,
        price_income_starting_values=[],
        endogenous_variable_defs: Dict[str, Callable[[Any, Any], float]] = None,
        neighborhoods=None,
        sample_alternatives=None,
        method="L-BFGS-B",
        minimize_options={},
        max_rent_to_income=None,
        household_housing_attributes=None,
        weights=None,
        max_chunk_bytes=2e9,
        est_first_stage_ses=True,
        seed=None,
        price_file=None,
    ):
        """
        Initialize a Tra sorting model

        :param housing_attributes: Attributes of housing choices. Price should not be an attribute here.
        :type housing_attributes: Pandas dataframe.

        :param household_attributes: Attributes of households. It is okay if income is an attribute here as long as it
            is also passed in separately - since we assume it doesn't change when a household moves.
        :type household_attributes: Pandas dataframe

        :param household_housing_attributes: Attributes of the combination of households and housing choices,
            multiindexed with household ID and choice ID. Must contain all possible combinations.
        :type household_housing_attributes: Pandas dataframe

        :param interactions: Which household attributes should be interacted with which housing attributes in the
            first stage. Endogenous variables may be included as housing attributes here.
        :type interactions: iterable of tuple (household_attributes column, housing_attributes column)

        :param second_stage_params: Which housing attributes to include in the second stage fit (decomposition of mean
            parameters). If None, don't fit a second stage (this stage is not necessary for simulation). Note that
            currently endogenous variables cannot be used in this stage.
        :type second_stage_params: iterable of string

        :param price: Observed price of each housing choice. This price should be over the same time period as income
            (for example, annualized prices with annual incomes)
        :type price: Pandas series of numbers, indexed like housing_attributes

        :param income: Household income
        :type income: Pandas series of numbers, indexed like household_attributes

        :param choice: The choice made by the household. Should be index values in household_attributes
        :param type: Pandas series, indexed like household_attributes

        :param endogenous_variable_defs: This is a dictionary mapping variable names to functions that calculate them,
            for endogenous variables that need to be updated in the sorting stage of the model, as household re-sort
            themselves across the region. The function should accept a data frame of housing attributes, a series with
            household income, and a numpy array of weights for how likely that household is to live in a particular
            neighborhood. Should return a float scalar for that neighborhood.
        :type endogenous_variable_defs: Dict of string -> function

        :param neighborhoods: Pandas series of which neighborhood each housing choice is in, for calculation of
            endogenous variables. Indexed like housing_attributes.
        :type neighborhoods: pd.Series

        :param sample_alternatives: If a positive integer, take a random sample of this size for available alternatives
            to estimate the model. Recommended in models with a large number of alternatives. If 0 or None, no sampling
                will be performed.
        :type sample_alternatives: int or None

        :param method: A scipy.optimize.minimize method used for maximizing log-likelihood of first stage, default
            'L-BFGS-B'
        :type method: str

        :param minimize_options: Options for selected minimizer
        :type minimize_options: dict


        :param max_rent_to_income: Maximum proportion of income rent can be for the alternative to be included in the
            choice set - in (0, 1] if the price/income transformation can't handle income less than rent
        :type max_rent_to_income: float or None

        :param max_chunk_bytes: Maximum number of bytes to use for a single chunk of the full alternatives array.
            The most memory-constrained part of the alogorithm is clearing the market, which requires computing the dot
            product of a n_households x n_housing_alternatives * n_unequilibrated_alternatives array and a coefficients
            array. However, since the coefficients array is a column vector, we can compute the product of chunks of the
            full alternatives array and the column vector, then concatenate.
            This tuning parameter controls how large these chunks are in bytes. Default 2GB.
        :type max_chunk_bytes: int

        :param est_first_stage_ses: If False, do not estimate standard errors for the first stage. This can
            significantly speed convergence and is recommended during model development.
        :type est_first_stage_ses: bool

        :param seed: seed for the random number generator used in sampling alternatives
        :type seed: int

        :param price_file: where prices are recorded in sorting
        :type price_file: str
        """
        self.housing_attributes = housing_attributes
        self.household_attributes = household_attributes
        self.household_housing_attributes = household_housing_attributes
        self.interactions = interactions
        self.second_stage_params = second_stage_params
        self.price = price.reindex(housing_attributes.index)
        self.orig_price = self.price.copy()
        self.income = income.reindex(household_attributes.index)
        self.choice = choice.reindex(household_attributes.index)
        self.unequilibrated_choice = unequilibrated_choice.reindex(
            household_attributes.index
        )
        self.unequilibrated_hh_params = unequilibrated_hh_params
        self.unequilibrated_hsg_params = unequilibrated_hsg_params
        self.sample_alternatives = sample_alternatives
        self.alternatives_stds = None

        if weights is None:
            self.weights = None
            self.weighted_supply = self.choice.value_counts()
        else:
            self.weights = weights.reindex(household_attributes.index)
            self.weighted_supply = self.weights.groupby(self.choice).sum()

        self.endogenous_variable_defs = endogenous_variable_defs
        if neighborhoods is not None:
            self.neighborhoods = neighborhoods.reindex(self.housing_attributes.index)
        else:
            self.neighborhoods = None

        self.price_income_transformation = price_income_transformation
        self.price_income_starting_values = price_income_starting_values
        self.method = method
        self.minimize_options = minimize_options
        self.max_rent_to_income = max_rent_to_income
        self.max_chunk_bytes = max_chunk_bytes
        self.est_first_stage_ses = est_first_stage_ses

        self.seed = seed
        self._rng = np.random.default_rng(seed=seed)

        self.validate()

        self.creation_time = datetime.datetime.today()

        assert (
            price_income_transformation.n_params == 0
        ), "Parameterized price_income_transformations not currently supported"

    def validate(self):
        # TODO this could easily be moved to a new file
        allPassed = True

        choiceCount = self.choice.value_counts().reindex(
            self.housing_attributes.index, fill_value=0
        )
        if not np.all(choiceCount > 0):
            choiceList = " - " + "\n - ".join(choiceCount.index[choiceCount == 0])
            LOG.error(
                f"Some housing alternatives are not chosen by any households!\n{choiceList}"
            )
            allPassed = False

        if self.max_rent_to_income is not None and not np.all(
            self.income.values * self.max_rent_to_income
            > self.price.loc[self.choice].values
        ):
            LOG.error(
                "Some households pay more in rent than the max rent to income ratio"
            )
            allPassed = False

        for hhattr, hsgattr in self.interactions:
            if hhattr not in self.household_attributes.columns:
                LOG.error(
                    f"Attribute {hhattr} is used in interactions but is not in household_attributes"
                )
                allPassed = False

            if self.household_attributes[hhattr].isnull().any():
                LOG.error(f"Attribute {hhattr} contains NaNs")
                allPassed = False

            if hsgattr not in self.housing_attributes.columns:
                if (
                    self.endogenous_variable_defs is None
                    or hsgattr not in self.endogenous_variable_defs
                ):
                    LOG.error(
                        f"Attribute {hsgattr} is used in interactions but is not in housing_attributes"
                    )
                    allPassed = False

            if (
                hsgattr in self.housing_attributes.columns
                and self.endogenous_variable_defs is not None
                and hsgattr in self.endogenous_variable_defs
            ):
                LOG.error(
                    f"{hsgattr} in both housing_attributes and endogenous variables"
                )
                allPassed = False

            if (
                hsgattr in self.housing_attributes.columns
                and self.housing_attributes[hsgattr].isnull().any()
            ):
                LOG.error(f"Attribute {hsgattr} contains NaNs")
                allPassed = False

        if self.second_stage_params is not None:
            for hsgattr in self.second_stage_params:
                if hsgattr not in self.housing_attributes.columns:
                    LOG.error(
                        f"Attribute {hsgattr} is used in second stage but is not in housing_attributes"
                    )
                    allPassed = False

                if self.housing_attributes[hsgattr].isnull().any():
                    LOG.error(f"Attribute {hsgattr} contains NaNs")
                    allPassed = False

        if self.endogenous_variable_defs is not None:
            if self.neighborhoods is None:
                LOG.error("Neighborhoods required when endogenous variables in use")
                allPassed = False
            if self.neighborhoods.isnull().any():
                LOG.error("All choices must have a neighborhood defined")
                allPassed = False

        # TODO more checks
        if allPassed:
            LOG.info("All validation checks passed!")
        else:
            raise ValueError("Some validation checks failed (see log messages)")

    def materialize_alternatives(
        self,
        hhidx,
        choiceidx,
        uneqchoiceidx,
        hh_hsgidx=None,
        price_income_params=None,
        materialize=True,
        include_budget=True,
    ):
        """
        Materialize the alternatives for hhidx, choiceidx, and uneqchoiceidx, and return them.

        These should be formatted like so, with hhidx changing slowest and uneqchoiceidx changing fastest.
        if there are three households, three housing choices, and three unequilibrated choices:
        hhidx:         0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2
        choiceidx:     0 0 0 1 1 1 2 2 2 0 0 0 1 1 1 2 2 2 0 0 0 1 1 1 2 2 2
        uneqchoiceidx: 0 1 2 0 1 2 0 1 2 0 1 2 0 1 2 0 1 2 0 1 2 0 1 2 0 1 2

        It is okay if they are not sequential, but they should be monotonically increasing.

        hh_hsgidx is the integer indices in household_housing_attributes for the selected household/housing combinations, same length
        as hhidx etc.

        If materialize is False, return a dask array rather than a materialized Numpy array
        """
        start_time = time.perf_counter()

        if price_income_params is None:
            # can't refer to self in function def
            price_income_params = self.price_income_starting_values

        assert len(hhidx) == len(choiceidx) and len(choiceidx) == len(uneqchoiceidx)

        if self.household_housing_attributes is not None:
            assert hh_hsgidx is not None and len(hhidx) == len(hh_hsgidx)

        LOG.info(f"materializing {len(hhidx)} choices")

        # first, create data for the interactions
        colnames = []

        # + 1 for budget param
        ncols = (
            len(self.interactions)
            + (len(self.unequilibrated_hh_params) + len(self.unequilibrated_hsg_params))
            * (len(self.unequilibrated_choice_xwalk) - 1)
            + 1
        )

        if self.household_housing_attributes is not None:
            ncols += len(self.household_housing_attributes.columns)

        alternatives = []

        # budget is first column, to make updates easier
        colnames.append("budget")

        if include_budget:
            alt_income = self.income.astype("float64").values[hhidx]
            alt_price = self.price.astype("float64").values[choiceidx]
            # don't calc buget for options not in choice set
            # it may throw an error (e.g. log(neg) for logdiff)
            if self.max_rent_to_income is not None:
                feasible_alts = alt_income * self.max_rent_to_income > alt_price
            else:
                feasible_alts = np.full(len(alt_income), True)

            LOG.info(f"{np.sum(feasible_alts)} options appear in choice sets")

            budget = np.full(len(hhidx), np.nan)
            budget[feasible_alts] = self.price_income_transformation.apply(
                alt_income[feasible_alts],
                alt_price[feasible_alts],
                *price_income_params,
            )
            assert not np.any(np.isnan(budget[feasible_alts]))  # should be no nans left
            alternatives.append(budget)
            del alt_income, alt_price, budget  # save memory
        else:
            alternatives.append(da.zeros_like(hhidx))

        for hh_attr, hsg_attr in self.interactions:
            if hsg_attr in self.housing_attributes.columns:
                # TODO lots of type conversion happening here. Could maybe refactor to do less.
                alternatives.append(
                    da.from_array(
                        self.household_attributes[hh_attr]
                        .astype("float64")
                        .values[hhidx]
                        * self.housing_attributes[hsg_attr]
                        .astype("float64")
                        .values[choiceidx]
                    )
                )
            elif hsg_attr in self.endogenous_varnames:
                endogenous_col = self.endogenous_varnames.index(hsg_attr)
                alternatives.append(
                    da.from_array(
                        self.household_attributes[hh_attr]
                        .astype("float64")
                        .values[hhidx]
                        * self.endogenous_variables[
                            self.nbhd_for_choice[choiceidx], endogenous_col
                        ]
                    )
                )
            else:
                raise KeyError(
                    f"{hsg_attr} is not a housing attribute, exogenous or endogenous"
                )
            colnames.append(f"{hh_attr}:{hsg_attr}")

        # now add the attributes for the unequilibrated choice
        for param in self.unequilibrated_hh_params:
            vals = self.household_attributes[param].astype("float64").values[hhidx]
            for uneqchoice in range(1, len(self.unequilibrated_choice_xwalk)):
                # fill all rows that are not for this unequilibrated choice with 0s
                alternatives.append(
                    da.from_array(np.choose(uneqchoiceidx == uneqchoice, [0, vals]))
                )
                colnames.append(
                    f"{param}:uneq_choice_{self.unequilibrated_choice_xwalk[self.unequilibrated_choice_xwalk == uneqchoice].index[0]}"
                )

        for param in self.unequilibrated_hsg_params:
            vals = self.housing_attributes[param].astype("float64").values[choiceidx]
            for uneqchoice in range(1, len(self.unequilibrated_choice_xwalk)):
                # fill all rows that are not for this unequilibrated choice with 0s
                alternatives.append(
                    da.from_array(np.choose(uneqchoiceidx == uneqchoice, [0, vals]))
                )
                colnames.append(
                    f"{param}:uneq_choice_{self.unequilibrated_choice_xwalk[self.unequilibrated_choice_xwalk == uneqchoice].index[0]}"
                )

        if self.household_housing_attributes is not None:
            for c in self.household_housing_attributes.columns:
                alternatives.append(
                    da.from_array(self.household_housing_attributes[c].to_numpy())
                )
                colnames.append(c)

        alternatives = da.stack(alternatives, axis=1)

        if self.alternatives_stds is None:
            self.alternatives_stds = da.std(alternatives, axis=0)

        alternatives /= self.alternatives_stds

        if materialize:
            alternatives = alternatives.compute()

        total_time = time.perf_counter() - start_time
        LOG.info(
            f"Materialized alternatives into {human_shape(alternatives.shape)} array in {human_time(total_time)}"
        )
        self.alternatives_colnames = (
            colnames  # hacky to set this every time but it never changes
        )

        return alternatives

    def create_alternatives(self):
        LOG.info("Creating alternatives")
        startTime = time.perf_counter()

        LOG.info("Converting pandas data to numpy")
        self.housing_xwalk = pd.Series(
            np.arange(len(self.housing_attributes)), index=self.housing_attributes.index
        )

        # we always have an unequilibrated choice to simplify coding, it is just only a single choice if not specified
        # good ol' mononomial logit model
        unequilibrated_choice = (
            self.unequilibrated_choice.copy()
            if self.unequilibrated_choice is not None
            else pd.Series(np.zeros(len(self.choice), index=self.choice.index))
        )
        unique_unequilibrated_choices = unequilibrated_choice.unique()
        self.unequilibrated_choice_xwalk = pd.Series(
            np.arange(len(unique_unequilibrated_choices)),
            index=unique_unequilibrated_choices,
        )
        self.hh_xwalk = pd.Series(
            np.arange(len(self.household_attributes)),
            index=self.household_attributes.index,
        )

        self.hh_hsg_choice = self.housing_xwalk.loc[
            self.choice.loc[self.hh_xwalk.index]
        ].values
        self.hh_unequilibrated_choice = self.unequilibrated_choice_xwalk.loc[
            self.unequilibrated_choice.loc[self.hh_xwalk.index]
        ].values

        if self.endogenous_variable_defs is not None:
            # neighborhood indices
            unique_neighborhoods = self.neighborhoods.unique()
            self.nbhd_xwalk = pd.Series(
                np.arange(len(unique_neighborhoods)), index=unique_neighborhoods
            )
            # neighborhood for each housing choice, neighborhood index for each neighborhood
            self.nbhd_for_choice = self.nbhd_xwalk.loc[
                self.neighborhoods.loc[self.housing_xwalk.index]
            ].values  # index is sorted
        else:
            self.nbhd_xwalk = self.nbhd_for_choice = None

        # index of each household in the full alternatives dataset
        # repeated for each alternative (housing choice)
        # so if there are three households, three housing choices, and three unequilibrated choices:
        # hhidx:         0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2
        # choiceidx:     0 0 0 1 1 1 2 2 2 0 0 0 1 1 1 2 2 2 0 0 0 1 1 1 2 2 2
        # uneqchoiceidx: 0 1 2 0 1 2 0 1 2 0 1 2 0 1 2 0 1 2 0 1 2 0 1 2 0 1 2
        # NB this could also be conceptualized as a four-dimensional array (household * housing choice * unequilibrated choice * variables)
        # but, uh, let's not
        LOG.info("Indexing full alternatives dataset")
        self.full_hhidx = np.repeat(
            np.arange(len(self.household_attributes)),
            len(self.housing_attributes) * len(unique_unequilibrated_choices),
        )
        self.full_choiceidx = np.repeat(
            np.tile(
                np.arange(len(self.housing_attributes)), len(self.household_attributes)
            ),
            len(unique_unequilibrated_choices),
        )
        self.full_uneqchoiceidx = np.tile(
            np.arange(len(unique_unequilibrated_choices)),
            len(self.housing_attributes) * len(self.household_attributes),
        )
        self.full_hsgchosen = self.hh_hsg_choice[self.full_hhidx] == self.full_choiceidx
        self.full_uneqchosen = (
            self.hh_unequilibrated_choice[self.full_hhidx] == self.full_uneqchoiceidx
        )
        self.full_chosen = self.full_hsgchosen & self.full_uneqchosen

        # create indices into the hh_housing_attributes so we can use it like a numpy array rather than with slow Pandas indexing
        # full_hh_hsg[i] is the 0-based index into household_housing_attributes for full_alternative i.
        # how we do it: build a Pandas series numbered 0...n and indexed like household_housing_attributes. then we zip the crosswalks together
        # and select from household_housing_attributes.
        if self.household_housing_attributes is not None:
            # do this in chunks to save memory
            self.full_hh_hsgidx = np.full_like(self.full_hhidx, -1, dtype="int32")
            hh_hsg_loc = pd.Series(
                np.arange(len(self.household_housing_attributes)),
                index=self.household_housing_attributes.index,
            )
            for chunk_start in range(0, len(self.full_hhidx), 100000):
                chunk_end = min(
                    chunk_start + 100000, len(self.household_housing_attributes)
                )
                self.full_hh_hsgidx[chunk_start:chunk_end] = hh_hsg_loc.loc[
                    list(
                        zip(
                            self.hh_xwalk.index[self.full_hhidx[chunk_start:chunk_end]],
                            self.housing_xwalk.index[
                                self.full_choiceidx[chunk_start:chunk_end]
                            ],
                        )
                    )
                ].values
            del hh_hsg_loc  # save memory
        else:
            self.full_hh_hsgidx = None

        # calculate endogenous variables if necessary (conditional is inside the function)
        self.initialize_or_update_endogenous_variables(initial=True)

        if self.sample_alternatives is None or self.sample_alternatives <= 0:
            if self.max_rent_to_income is None:
                self.alternatives = self.materialize_alternatives(
                    self.full_hhidx,
                    self.full_choiceidx,
                    self.full_uneqchoiceidx,
                    self.full_hh_hsgidx
                    if self.household_housing_attributes is not None
                    else None,
                )
                self.alternatives_hhidx = self.full_hhidx
                self.alternatives_choiceidx = self.full_choiceidx
                self.alternatives_uneqchoiceidx = self.full_uneqchoiceidx
                self.alternatives_hsgchosen = self.full_hsgchosen
                self.alternatives_uneqchosen = self.full_uneqchosen
                self.alternatives_chosen = self.full_chosen
            else:
                # TODO
                raise ValueError("max_rent_to_income with no sampling is unimplemented")

        else:
            LOG.info("Sampling alternatives")
            if self.max_rent_to_income is None:
                # note that we do not include the other unequilibrated choises for the chosen housing unit here, so they are not selected
                # randomly. We are randomly sampling housing alternatives, but always use all unequilibrated alternatives.
                feasible_unchosen_alts = ~self.full_hsgchosen
            else:
                feasible_unchosen_alts = (
                    self.income.astype("float64").values[self.full_hhidx]
                    * self.max_rent_to_income
                    > self.price.astype("float64").values[self.full_choiceidx]
                ) & ~self.full_hsgchosen

            # unequilibrated alternatives are not sampled
            n_housing_alts_per_hh = np.bincount(
                self.full_hhidx[feasible_unchosen_alts]
            ) / len(self.unequilibrated_choice_xwalk)

            def random_sel(n):
                if n <= self.sample_alternatives - 1:
                    return np.repeat([True], n)
                else:
                    ret = np.arange(n) < self.sample_alternatives - 1
                    self._rng.shuffle(ret)
                    return ret

            # since households are the outermost index, and unequilibrated choices are the innermost index, we can get away with this
            # the repeat() makes sure all unquilibrated alternatives are selected (recall they are always adjacent), and concatenating
            # is correct since households are the slowest-changing index
            sampled_mask = np.concatenate(
                [
                    np.repeat(random_sel(n), len(self.unequilibrated_choice_xwalk))
                    for n in n_housing_alts_per_hh
                ]
            )

            unchosen_sampled_idxs = np.arange(len(self.full_hhidx))[
                feasible_unchosen_alts
            ][sampled_mask]
            del sampled_mask
            chosen_idxs = np.arange(len(self.full_hhidx))[
                self.full_hsgchosen
            ]  # we do not sample uneq alternatives

            sampled_idxs = np.concatenate([unchosen_sampled_idxs, chosen_idxs])
            # put them back in the household > housing > uneq order
            np.sort(sampled_idxs)

            self.alternatives_hhidx = self.full_hhidx[sampled_idxs]
            self.alternatives_choiceidx = self.full_choiceidx[sampled_idxs]
            self.alternatives_uneqchoiceidx = self.full_uneqchoiceidx[sampled_idxs]
            self.alternatives_hsgchosen = self.full_hsgchosen[sampled_idxs]
            self.alternatives_uneqchosen = self.full_uneqchosen[sampled_idxs]
            self.alternatives_chosen = self.full_chosen[sampled_idxs]
            if self.household_housing_attributes is not None:
                self.alternatives_hh_hsgidx = self.full_hh_hsgidx[sampled_idxs]
            else:
                self.alternatives_hh_hsgidx = None
            self.alternatives = self.materialize_alternatives(
                self.alternatives_hhidx,
                self.alternatives_choiceidx,
                self.alternatives_uneqchoiceidx,
                self.alternatives_hh_hsgidx
                if self.household_housing_attributes is not None
                else None,
            )

        endTime = time.perf_counter()
        LOG.info(
            f"Created alternatives for {len(self.household_attributes)} households in {endTime - startTime:.3f} seconds"
        )
        LOG.info(f"Alternatives dimensions: {human_shape(self.alternatives.shape)}")
        LOG.info(f"Alternatives use {human_bytes(self.alternatives.nbytes)} memory")

    def full_utility(self, include_budget=True, include_ascs=True):
        """
        Calculate full utilities (i.e. utilities for all alternatives, not just sampled ones).
        """
        util_start_time = time.perf_counter()

        # convert supply to an np array
        lnsupply = np.log(self.weighted_supply.loc[self.housing_xwalk.index].values)

        if self.price_income_transformation.n_params > 0:
            coefs = self.first_stage_fit.params.values[
                : -self.price_income_transformation.n_params
            ]
            price_income_params = self.first_stage_fit.params.values[
                -self.price_income_transformation.n_params :
            ]
        else:
            coefs = self.first_stage_fit.params.values
            price_income_params = np.zeros(0)

        alts = self.materialize_alternatives(
            self.full_hhidx,
            self.full_choiceidx,
            self.full_uneqchoiceidx,
            self.full_hh_hsgidx,
            price_income_params=price_income_params,
            materialize=False,
            include_budget=include_budget,
        )

        # add systematic utility and deterministic part of ASC based on market share
        # TODO okay to just add log(weighted) here when ASCs were calc'd with log(unweighted)? I think so.
        utility = da.dot(alts, coefs) + lnsupply[self.full_choiceidx]

        # compute now so that we're not holding on to big arrays in dask before calling compute
        utility = utility.compute()

        if include_ascs:
            # do this in two steps to save memory
            utility += self.first_stage_ascs.values[self.full_choiceidx]
            utility += self.first_stage_uneq_ascs.values[self.full_uneqchoiceidx]

        util_end_time = time.perf_counter()
        LOG.info(
            f"Computing full utility took {human_time(util_end_time - util_start_time)}"
        )
        return utility

    def fit_first_stage(self):
        LOG.info("fitting first stage")

        self._log_supply = np.log(np.bincount(self.hh_hsg_choice))[
            self.alternatives_choiceidx
        ]

        self.first_stage_fit = MNLFullASC(
            alternatives=self.alternatives,
            choiceidx=(self.alternatives_choiceidx, self.alternatives_uneqchoiceidx),
            hhidx=self.alternatives_hhidx,
            chosen=self.alternatives_chosen,
            supply=(
                np.bincount(self.hh_hsg_choice),
                np.bincount(self.hh_unequilibrated_choice),
            ),
            starting_values=np.zeros(self.alternatives.shape[1], dtype="float64"),
            param_names=[
                *self.alternatives_colnames,
                *self.price_income_transformation.param_names,
            ],
            method=self.method,
            minimize_options=self.minimize_options,
            est_ses=self.est_first_stage_ses,
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
            feasible_alts = (
                self.income.astype("float64").values[self.full_hhidx]
                * self.max_rent_to_income
                > self.price.astype("float64").values[self.full_choiceidx]
            )

        base_utility = self.full_utility(include_ascs=False)

        assert not np.any(np.isnan(base_utility[feasible_alts]))

        fullAscStartTime = time.perf_counter()

        ascs = compute_ascs(
            base_utilities=base_utility[feasible_alts],
            supply=(
                # of homes
                self.weighted_supply.loc[self.housing_xwalk.index].values,
                # of unequilibrated choices
                # While we don't (obviously) equilibrate unequilibrated choices, we do use their supply to find their ASCs in the first-stage fit
                np.bincount(
                    self.hh_unequilibrated_choice,
                    self.weights.loc[self.hh_xwalk.index].values,
                ),
            ),
            hhidx=self.full_hhidx[feasible_alts],
            choiceidx=(
                self.full_choiceidx[feasible_alts],
                self.full_uneqchoiceidx[feasible_alts],
            ),
            starting_values=self.first_stage_fit.ascs,
            weights=self.weights.loc[self.hh_xwalk.index].values,
            log=True,
        )
        fullAscEndTime = time.perf_counter()

        # descale coefs
        # but don't descale transformation parameters
        # recall that the _result_ of the transformation, not the inputs, is what is scaled, so this is okay
        # TODO make sure this is rescaling correctly
        scalars = pd.Series(self.alternatives_stds, index=self.alternatives_colnames)
        scalars.loc[self.price_income_transformation.param_names] = 1
        scalars = scalars.reindex(self.first_stage_fit.params.index)
        self.first_stage_fit.params /= scalars
        if self.est_first_stage_ses and self.first_stage_fit.se is not None:
            self.first_stage_fit.se /= scalars

        # no scaling needed anymore
        self.alternatives_stds = np.ones_like(self.alternatives_stds)

        self.first_stage_ascs = pd.Series(ascs[0], index=self.housing_xwalk.index)
        self.first_stage_uneq_ascs = pd.Series(
            ascs[1], index=self.unequilibrated_choice_xwalk.index
        )
        LOG.info(
            f"Finding full ASCs took {human_time(fullAscEndTime - fullAscStartTime)}"
        )

        # don't serialize
        self.first_stage_fit.alternatives = None
        self.first_stage_fit._log_supply = None

    def fit_second_stage(self):
        LOG.info("fitting second stage")
        startTime = time.perf_counter()
        second_stage_exog = sm.add_constant(
            self.housing_attributes[self.second_stage_params]
        )
        second_stage_endog = self.first_stage_ascs.reindex(second_stage_exog.index)

        mod = sm.OLS(second_stage_endog, second_stage_exog)
        self.second_stage_fit = mod.fit()
        self.type_shock = self.second_stage_fit.resid
        endTime = time.perf_counter()
        LOG.info(f"Fit second stage in {endTime - startTime:.2f} seconds")

    def fit(self):
        self.fit_first_stage()
        if self.second_stage_params is not None:
            self.fit_second_stage()
        else:
            LOG.info("No second stage requested")

    def sort(self, maxiter=np.inf):
        """
        Clear the market with a change to supply

        TODO document which data structures can be changed and still have this function return correct results
        """
        LOG.info("Clearing the market and sorting households")

        # convert supply to an np array
        supply = self.weighted_supply.loc[self.housing_xwalk.index].values

        # allow for slight floating point error several orders of magnitude smaller than convergence criterion
        if np.abs(np.sum(supply) - np.sum(self.weights)) > 1e-8:
            raise ValueError(
                f"total supply has changed! expected {np.sum(self.weights)} but found {np.sum(supply)}"
            )

        # first update second stage
        if self.second_stage_params is not None:
            LOG.info("updating second stage")
            pred_ascs = (
                self.second_stage_fit.predict(
                    sm.add_constant(self.housing_attributes[self.second_stage_params])
                )
                + self.type_shock
            )

            maxabsdiff = np.max(np.abs(pred_ascs - self.first_stage_ascs))
            LOG.info(
                f"Second stage updated with changes to first-stage ASCs of up to {maxabsdiff:.2f}"
            )
            self.first_stage_ascs = pred_ascs
        else:
            LOG.info("No second stage fit, not updating")
            pred_ascs = self.first_stage_ascs[0]

        if self.price_income_transformation.n_params > 0:
            price_income_params = self.first_stage_fit.params.values[
                -self.price_income_params :
            ]
        else:
            price_income_params = np.zeros(0)

        itr = 0
        startTimeClear = time.perf_counter()
        all_prices = []

        while True:
            if itr > maxiter:
                LOG.error(f"Prices FAILED TO CONVERGE after {itr} iterations")
                break
            itr += 1
            LOG.info(f"sorting: begin iteration {itr}")
            self.initialize_or_update_endogenous_variables(initial=False)

            LOG.info("finding non-price utilites")
            non_price_utilities = self.full_utility(include_budget=False)
            assert not np.any(np.isnan(non_price_utilities))

            # Note that I am intentionally _not_ dropping hh/choice combinations here that do not meet rent to income criteria, because which households those are
            # might change in the sorting phase. The filtering happens there. This does not matter for the calculation of utility below, since utilities for
            # different alternatives are independent of each other and the budget is set to zero anyhow.

            new_prices, converged = clear_market_iter(
                non_price_utilities=non_price_utilities,
                hhidx=self.full_hhidx,
                choiceidx=self.full_choiceidx,
                supply=supply,
                income=self.income.loc[self.hh_xwalk.index].values,
                starting_price=self.price.loc[self.housing_xwalk.index].values,
                price_income_transformation=self.price_income_transformation,
                price_income_params=price_income_params,
                budget_coef=self.first_stage_fit.params["budget"],
                max_rent_to_income=self.max_rent_to_income,
                weights=self.weights.loc[self.hh_xwalk.index].values
                if self.weights is not None
                else None,
            )

            all_prices.append(new_prices)

            if self.price_file is not None:
                np.save(self.price_file, np.array(all_prices), allow_pickle=False)

            assert np.all(new_prices > 0), "some prices are 0 or less!"

            new_prices = pd.Series(new_prices, index=self.housing_xwalk.index)
            self.price = new_prices

            if converged:
                LOG.info(f"prices converged after {itr} iterations")
                break

        endTimeClear = time.perf_counter()
        LOG.info(f"sorting took {human_time(endTimeClear - startTimeClear)}")

    def initialize_or_update_endogenous_variables(self, initial: bool) -> None:
        """
        Initialize endogenous variables, or update them based on predicted probabilities after the model has been fit.
        :param initial: if True, will compute endogenous variables based on observed choices. If False, will compute
            based on predicted probabilities
        :type initial: bool
        """
        if self.endogenous_variable_defs is None:
            LOG.info("not updating endogenous variables as none are defined")
        else:
            if initial:
                # initial fit: use actual chosen values to compute endogenous variables
                LOG.info("creating endogenous variables")
                weighted_probs = self.full_hsgchosen
            else:
                # if model is already fitted, update endogenous variables with fitted values
                LOG.info("updating endogenous variables")
                weighted_probs = self._probabilities()

            if self.weights is not None:
                # if weights are present, use them
                weighted_probs = (
                    weighted_probs
                    * self.weights.loc[self.hh_xwalk.index].values[self.full_hhidx]
                )

            # if this is the initial run, initialize endogenous_varnames
            # need to materialize and save because dict.keys() order is not guaranteed, though in recent versions of
            # Python it is stable (in fact, may have always been stable?)
            if initial:
                self.endogenous_varnames = list(self.endogenous_variable_defs.keys())

            # Neighborhood for every element of weighted_probs
            nbhdidx = self.nbhd_for_choice[self.full_choiceidx]

            # reset endogenous variables. this creates a new array. if we instead cleared the old array, we would need
            # to create a copy in the if block below to ensure that the original values were preserved for comparison
            # purposes
            self.endogenous_variables = np.full(
                (np.max(self.nbhd_for_choice) + 1, len(self.endogenous_varnames)),
                np.nan,
                "float64",
            )

            # compute endogenous variables
            for neighborhood in range(np.max(self.nbhd_for_choice) + 1):
                nbhdmask = nbhdidx == neighborhood
                # all probabilities for the neighborhood
                nbhdweights = weighted_probs[nbhdmask]
                # how likely every household is to choose this neighborhood
                hhweights = np.bincount(self.full_hhidx[nbhdmask], weights=nbhdweights)
                # use the precomputed list to avoid order changing, as iteration order in python dicts is not guaranteed
                # endogenous_varnames is created by initialize_endogenous_variables
                for i, varname in enumerate(self.endogenous_varnames):
                    func = self.endogenous_variable_defs[varname]
                    self.endogenous_variables[neighborhood, i] = func(
                        self.household_attributes, self.income, hhweights
                    )

            # check our work
            assert not np.any(
                np.isnan(self.endogenous_variables)
            ), "some endogenous variables are nan!"

            if initial:
                # save the initial variables for later comparison
                # this array is never modified, an entirely new array is created when sorting, so no need to copy
                self.orig_endogenous_variables = self.endogenous_variables

    def _probabilities(self):
        "Compute probabilities and return as numpy array, use .probabilities() for a Pandas data frame"
        LOG.info("finding utility")

        if self.max_rent_to_income is None:
            feasible_alts = np.full(True, len(self.full_hhidx))
        else:
            feasible_alts = (
                self.income.astype("float64").values[self.full_hhidx]
                * self.max_rent_to_income
                > self.price.astype("float64").values[self.full_choiceidx]
            )

        exp_utility = np.exp(self.full_utility()[feasible_alts])
        assert not np.any(np.isnan(exp_utility))
        expsums = np.bincount(self.full_hhidx[feasible_alts], exp_utility)
        probs = np.zeros_like(self.full_choiceidx, dtype="float64")
        # alts not in choice set b/c infeasible are left with probability zero
        probs[feasible_alts] = exp_utility / expsums[self.full_hhidx[feasible_alts]]
        return probs

    def probabilities(self):
        "Return choice probabilities as Pandas dataframe, with columns for household ID, choice ID, and uneq choice ID"
        probs = self._probabilities()
        return pd.DataFrame(
            {
                "hh": self.hh_xwalk.index[self.full_hhidx],
                "housing": self.housing_xwalk.index[self.full_choiceidx],
                "uneq_choice": self.unequilibrated_choice_xwalk.index[
                    self.full_uneqchoiceidx
                ],
                "probability": probs,
            }
        )

    def _mkt_shares(self):
        probs = (
            self._probabilities()
            * self.weights.loc[self.hh_xwalk.index].values[self.full_hhidx]
        )
        return np.bincount(self.full_choiceidx, probs)

    def mkt_shares(self):
        return pd.Series(self._mkt_shares(), index=self.housing_xwalk.index)

    def _uneq_mkt_shares(self):
        probs = (
            self._probabilities()
            * self.weights.loc[self.hh_xwalk.index].values[self.full_hhidx]
        )
        return np.bincount(self.full_uneqchoiceidx, probs)

    def uneq_mkt_shares(self):
        return pd.Series(
            self._uneq_mkt_shares(), index=self.unequilibrated_choice_xwalk.index
        )

    def savenew(self, basefile):
        save_load.save(basefile, self)

    @classmethod
    def loadnew(cls, basefile):
        return save_load.load(basefile)

    def to_text(self, fn=None):
        "Save model results as text. If fn==None, return as string"

        outstring = """Equilibrium sorting model (Tra [2007] formulation, price in first stage as budget constraint)
Model run initiated at {creation_time}
Budget function {price_income_transformation}

First stage (discrete choice sorting model):
{first_stage_summary}

Equilibrated (housing) ASCs:
{equilibrated_ascs}

Unequilibrated ASCs:
{unequilibrated_ascs}

Second stage (OLS parameters):
{second_stage_summary}

Fit with EqSorMo version {version}, https://github.com/mattwigway/eqsormo
        """.format(
            first_stage_summary=self.first_stage_fit.summary(),
            unequilibrated_ascs=pd.DataFrame(self.first_stage_uneq_ascs).to_string(),
            equilibrated_ascs=pd.DataFrame(
                self.first_stage_ascs.describe()
            ).to_string(),
            second_stage_summary=self.second_stage_fit.summary(),
            creation_time=self.creation_time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            price_income_transformation=self.price_income_transformation.name,
            version=eqsormo.version,
        )

        if fn is not None:
            with open(fn, "w") as outfile:
                outfile.write(outstring)
        else:
            return outstring

    # don't pickle fullAlternatives
    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k != "fullAlternatives"}

    @classmethod
    def from_pickle(cls, fn):
        tra = super().from_pickle(fn)
        tra.create_full_alternatives()
        return tra
