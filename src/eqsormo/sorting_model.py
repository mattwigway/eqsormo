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

# TODO Do we need this file anymore?

from logging import getLogger
import time
import statsmodels.api as sm
import pandas as pd
import numpy as np
import scipy.optimize, scipy.stats
import linearmodels
from tqdm import tqdm
import pickle
from .ascs import compute_ascs

LOG = getLogger(__name__)


class SortingModel(object):
    """
    Represents a random utility based equilibrium sorting model.

    Based on the model described in Klaiber and Phaneuf (2010) "Valuing open space in a residential sorting model of the Twin Cities"
    https://doi.org/10.1016/j.jeem.2010.05.002 (no open access version available, unfortunately)
    """

    def __init__(
        self,
        altCharacteristics,
        altPrice,
        hh,
        hhChoice,
        interactions,
        initialPriceCoef,
        sampleAlternatives=None,
        method="bfgs",
    ):
        """
        :param altCharacteristics: Home and neighborhood attributes of housing alternatives/types (assumed that all alternatives are available to all households)
        :type altHousing: Pandas dataframe
        
        :param altPrice: Equilibrium (non-instrumented) price of housing alternatives (will be replaced with instrumented version)
        :type altPrice: Pandas series of float64, indexed like altHousing and altNeighborhood

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
        self.altCharacteristics = altCharacteristics.astype("float64").copy()
        self.price = altPrice.copy()
        self.hh = hh.copy().apply(
            lambda col: col - col.mean()
        )  # predemean household characteristics
        self.hh.index.name = "household"  # for joining convenience later
        self.hhChoice = hhChoice.copy()
        self.interactions = interactions
        self.initialPriceCoef = initialPriceCoef
        self.sampleAlternatives = sampleAlternatives

        #: supply by housing type
        self.supply = self.hhChoice.value_counts().astype("float64")

        self.validate()

        assert not "price" in self.altCharacteristics.columns
        self.method = method

    def validate(self):
        "Check for obvious errors in model inputs"
        LOG.warn(
            "Validation function not implemented. You better be darn sure of your inputs."
        )

    def fit(self):
        "Fit the whole model"
        startTime = time.perf_counter()
        LOG.info("Fitting equilibrium sorting model")
        if self.alternatives is None:
            raise RuntimeError("Alternatives not yet created")

        self.fit_first_stage()
        self.fit_second_stage()
        endTime = time.perf_counter()
        LOG.info(
            f"""
Fitting sorting model took {endTime - startTime:.3f} seconds.
Convergence:
  First stage: {self.first_stage_converged}
"""
        )

    def create_alternatives(self):
        LOG.info("Creating alternatives")
        startTime = time.perf_counter()

        self.fullAlternatives = pd.concat(
            [self.altCharacteristics for i in range(len(self.hh))], keys=self.hh.index
        )
        self.fullAlternatives["chosen"] = False
        self.fullAlternatives["hhchoice"] = self.hhChoice.reindex(
            self.fullAlternatives.index, level=0
        )
        self.fullAlternatives.loc[
            self.fullAlternatives.index.get_level_values(1)
            == self.fullAlternatives.hhchoice,
            "chosen",
        ] = True

        LOG.info("created full set of alternatives, now sampling if requested")

        if self.sampleAlternatives <= 0 or self.sampleAlternatives is None:
            self.alternatives = self.fullAlternatives
        else:
            unchosenAlternatives = (
                self.fullAlternatives[~self.fullAlternatives.chosen]
                .groupby(level=0)
                .apply(lambda x: x.sample(self.sampleAlternatives - 1))
            )
            unchosenAlternatives.index = unchosenAlternatives.index.droplevel(
                0
            )  # fix dup'd household level due to groupby
            self.alternatives = pd.concat(
                [
                    unchosenAlternatives,
                    self.fullAlternatives[self.fullAlternatives.chosen],
                ]
            ).sort_index(level=[0, 1])

        self.alternatives.drop(columns=["chosen"], inplace=True)
        self.fullAlternatives.drop(columns=["chosen"], inplace=True)

        endTime = time.perf_counter()
        LOG.info(
            f"Created {len(self.alternatives)} alternatives for {len(self.hh)} in {endTime - startTime:.3f} seconds"
        )

    def first_stage_utility(self, params, mean_indirect_utility):
        # TODO I don't think that adding the mean_indirect_utility this way will work from an indexing standpoint
        # diffs is differences due to sociodemographics from mean indirect utility
        diffs = self.firstStageData.multiply(params, axis="columns").sum(axis="columns")
        utilities = (
            diffs
            + mean_indirect_utility.loc[
                self.firstStageData.index.get_level_values("choice")
            ].values
        )
        return utilities

    def first_stage_probabilities(self, params, mean_indirect_utility):
        utility = self.first_stage_utility(params, mean_indirect_utility)
        expUtility = np.exp(utility)
        if not np.all(np.isfinite(expUtility)):
            raise ValueError(
                f"Household/choice combinations {expUtility.index[~np.isfinite(expUtility)]} have non-finite utilities!"
            )
        return expUtility / expUtility.groupby(level=0).sum()

    def compute_mean_indirect_utility(self, params):
        # These are the alternative specific constants, which are not fit by ML, but rather using a contraction mapping that lets
        # the model converge faster - see Equation 16 of Bayer et al. (2004).
        mean_indirect_utility = compute_ascs(
            self.firstStageData, params, self.supply, self._prev_mean_indirect_utility
        )
        self._prev_mean_indirect_utility = mean_indirect_utility
        return mean_indirect_utility

    def first_stage_neg_loglikelihood(self, params):
        mean_indirect_utility = self.compute_mean_indirect_utility(params)
        logprobs = np.log(self.first_stage_probabilities(params, mean_indirect_utility))
        return -np.sum(logprobs.loc[list(zip(self.hhChoice.index, self.hhChoice))])

    def fit_first_stage(self):
        "Perform the first stage estimation"
        LOG.info("Performing first-stage estimation")

        startTime = time.perf_counter()

        self.firstStageData = pd.DataFrame()

        # demean sociodemographics so ASCs are interpretable as mean indirect utility (or something like that... TODO check)
        altsWithHhCharacteristics = self.alternatives.join(
            self.hh
        )  # should project to all alternatives TODO check

        for interaction in self.interactions:
            self.firstStageData[f"{interaction[0]}_{interaction[1]}"] = (
                altsWithHhCharacteristics[interaction[0]]
                * altsWithHhCharacteristics[interaction[1]]
            )

        # solve scaling issues
        stdevs = self.firstStageData.apply(np.std)
        self.firstStageData = self.firstStageData.divide(stdevs, axis="columns")

        LOG.info(f"Fitting {len(self.firstStageData.columns)} interaction parameters")

        self._prev_mean_indirect_utility = pd.Series(
            np.zeros(len(self.altCharacteristics)), index=self.altCharacteristics.index
        )

        self.first_stage_loglik_constants = -self.first_stage_neg_loglikelihood(
            np.zeros(len(self.firstStageData.columns))
        )

        minResults = scipy.optimize.minimize(
            self.first_stage_neg_loglikelihood,
            np.zeros(len(self.firstStageData.columns)),
            method=self.method,
            options={"disp": True},
        )

        self.first_stage_loglik_beta = -self.first_stage_neg_loglikelihood(minResults.x)
        self.interaction_params = (
            pd.Series(minResults.x, self.firstStageData.columns) / stdevs
        )  # correct the scaling
        self.interaction_params_se = (
            pd.Series(
                np.sqrt(np.diag(minResults.hess_inv)), self.firstStageData.columns
            )
            / stdevs
        )
        # TODO robust SEs
        self.mean_indirect_utility = self.compute_mean_indirect_utility(
            self.interaction_params
        )
        self.first_stage_converged = minResults.success

        endTime = time.perf_counter()
        if self.first_stage_converged:
            LOG.info(
                f"First stage converged in {endTime - startTime:.3f} seconds: {minResults.message}"
            )
        else:
            LOG.error(
                f"First stage FAILED TO CONVERGE in {endTime - startTime:.3f} seconds: {minResults.message}"
            )

    def fit_second_stage(self):
        "Fit the instrumental variables portion of the model"
        LOG.info("Fitting second stage")

        startTime = time.perf_counter()

        priceCoef = prevPriceCoef = self.initialPriceCoef

        iter = 0
        with tqdm() as pbar:
            while True:
                residual_utility = self.mean_indirect_utility - priceCoef * self.price
                # Constant should be included since location of ASCs is arbitrary, see Klaiber and Kuminoff (2014) note 9
                olsreg = sm.OLS(
                    residual_utility, sm.add_constant(self.altCharacteristics)
                )
                olsfit = olsreg.fit()

                # compute the price instrument by solving for prices that clear the market - which means the prices that produce the mean indirect utilities found
                # in the first stage, since those are the market-clearing utilities.
                priceIv = (self.mean_indirect_utility - olsfit.fittedvalues) / priceCoef

                ivreg = linearmodels.IV2SLS(
                    self.mean_indirect_utility,
                    sm.add_constant(self.altCharacteristics),
                    pd.DataFrame(self.price.rename("price")),
                    pd.DataFrame(priceIv.rename("price_iv")),
                )
                self.second_stage_fit = ivreg.fit()

                priceCoef = self.second_stage_fit.params.price

                iter += 1
                pbar.update()
                if np.abs(priceCoef - prevPriceCoef) < 1e-6:
                    endTime = time.perf_counter()
                    LOG.info(
                        f"Price coefficient converged in {endTime-startTime:.3f} seconds after {iter} iterations"
                    )
                    self.mean_params = self.second_stage_fit.params
                    self.mean_params_se = (
                        self.second_stage_fit.std_errors
                    )  # NB these are wrong b/c they don't account for variation in theta
                    self.type_shock = (
                        self.second_stage_fit.resids
                    )  # TODO ensure this does not include residuals from the first stage 2SLS estimates
                    self.price_iv = priceIv
                    break
                else:
                    prevPriceCoef = priceCoef

    def summary(self):
        # summarize params
        summary = pd.DataFrame(
            {
                "coef": pd.concat([self.interaction_params, self.mean_params]),
                "se": pd.concat(
                    [self.interaction_params_se, self.mean_params_se]
                ),  # TODO no standard errors for second stage yet...
            }
        )

        summary["z"] = summary.coef / summary.se
        # TODO should probably be t-test, but then I have to do a bunch of df calculations...
        summary["p"] = (1 - scipy.stats.norm.cdf(np.abs(summary.z))) * 2
        return summary

    def rebuild_full_alternatives_with_interactions(self):
        "Call after attributes of alternatives have changed to get utilities right"
        self.fullAlternativesWithInteractions = self.fullAlternatives.join(self.hh)

        for left, right in self.interactions:
            self.fullAlternativesWithInteractions[f"{left}_{right}"] = (
                self.fullAlternativesWithInteractions[left]
                * self.fullAlternativesWithInteractions[right]
            )

        self.fullTypeShock = self.type_shock.reindex(
            self.fullAlternativesWithInteractions.index, level=1
        )
        self.fullAlternativesWithInteractions["price"] = self.price.reindex(
            self.fullAlternativesWithInteractions.index, level=1
        )

    def utilities(self):
        "Get utilities of every household choosing every house type"
        # compute utilities, without unobserved price shocks
        params = pd.concat([self.mean_params, self.interaction_params])
        utilities = (
            sm.add_constant(self.fullAlternativesWithInteractions)[params.index]
            .multiply(params, "columns")
            .sum(axis="columns")
        )
        utilities += self.fullTypeShock

        # solve numerical problems by making all utilities positive
        # you can add a constant to all utilities for a particular household and get the same probabilities
        # utilities -= utilities.groupby(level=0).min().reindex(utilities.index, level=0)

        return utilities

    # work in percentages rather than probabilities to avoid underflow issues
    def probabilities(self, utilities=None):
        if utilities is None:
            utilities = self.utilities()

        expUtilities = np.exp(utilities)
        if not np.all(np.isfinite(expUtilities)):
            LOG.warn("some utilities are not finite when exponentiated")
        if not np.all(expUtilities > 0):
            LOG.warn("some utilities are zero when exponentiated")
        logsums = expUtilities.groupby(level=0).sum()  # group by households and sum
        percentages = (expUtilities) / logsums.reindex(expUtilities.index, level=0)
        return percentages

    def market_shares(self, utilities=None):
        return self.probabilities(utilities=utilities).groupby(level=1).sum()

    def clear_market(self):
        "Adjust prices so the market clears"
        prices = self.price  # for comparison later

        LOG.info("Clearing the market (everyone stand back)")

        # utilities with constants - pre compute for performance
        mean_indirect_utility = compute_ascs(
            self.fullAlternativesWithInteractions,
            self.interaction_params,
            self.supply,
            self.mean_indirect_utility,
        )

        # solve for price
        nonPriceParams = self.mean_params.drop(["price"])
        nonPriceObsUtilities = (
            sm.add_constant(self.altCharacteristics)[nonPriceParams.index]
            .multiply(nonPriceParams, axis="columns")
            .sum(axis="columns")
        )
        self.price = (
            mean_indirect_utility - nonPriceObsUtilities - self.type_shock
        ) / self.mean_params.price
        self.fullAlternativesWithInteractions["price"] = self.price.reindex(
            self.fullAlternativesWithInteractions.index, level=1
        )

        maxPriceChange = np.max(np.abs(self.price - prices))
        LOG.info(
            f"Market cleared with absolute price changes of up to {maxPriceChange}"
        )
        LOG.info(f"New price distribution: {self.price.describe()}")

    def to_pickle(self, fn):
        "Save a fit model to disk"
        if isinstance(fn, str):
            with open(fn, "wb") as out:
                pickle.dump(self, out)
        else:
            pickle.dump(self, fn)

    @staticmethod
    def from_pickle(fn):
        "Read a previously saved model"
        if isinstance(fn, str):
            with open(fn, "rb") as inf:
                model = pickle.load(inf)
        else:
            model = pickle.load(fn)

        if not isinstance(model, SortingModel):
            raise ValueError("File does not contain a sorting model!")

        return model
