#    Copyright 2020 Matthew Wigginton Conway

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# Author: Matthew Wigginton Conway <matt@indicatrix.org>,
#         School of Geographical Sciences and Urban Planning
#         Arizona State University

import pandas as pd
import numpy as np

import eqsormo
from eqsormo.tra import TraSortingModel

eqsormo.enable_logging()


def test_endogenous_variables(monkeypatch):
    model = TraSortingModel(
        housing_attributes=pd.DataFrame({"age": [10, 20, 30, 40, 50]}),
        household_attributes=pd.DataFrame({"white": [1, 1, 0, 0, 1]}),
        interactions=(("white", "prop_white"),),
        unequilibrated_hh_params=tuple(),
        unequilibrated_hsg_params=tuple(),
        second_stage_params=tuple(),
        # avoid price-income issues by making all prices less than all incomes - or maybe this should be tested
        # explicitly?
        price=pd.Series([5, 5, 5, 5, 5]),
        income=pd.Series([10, 20, 30, 40, 50]),
        choice=pd.Series([0, 1, 2, 3, 4]),
        unequilibrated_choice=pd.Series([0, 0, 0, 0, 0]),
        endogenous_variable_defs={
            "prop_white": lambda hh, inc, weights: np.average(
                hh.white, weights=weights
            ),
            "mean_income": lambda hh, inc, weights: np.average(inc, weights=weights),
        },
        neighborhoods=pd.Series(["a", "a", "b", "b", "b"]),
        fixed_price=0,
    )

    model.create_alternatives()

    model.initialize_or_update_endogenous_variables(initial=True)

    # as of Python 3.something, dict iteration order is the same as definition order. This test relies on this;
    # otherwise, we will not compare prop_white to expected prop_white. Can be fixed by actually reading the values in
    # endogenous_varnames
    assert model.endogenous_varnames == [
        "prop_white",
        "mean_income",
    ], "endogenous varnames incorrect or not in expected order"

    # as before, this could be harmless - but if our assumption in testing here was not met, later asserts would fail
    # in weird and wonderful ways
    assert np.all(
        model.nbhd_xwalk == pd.Series([0, 1], index=["a", "b"])
    ), "nbhd_xwalk not in expected order"

    # again, check that order is as expected - changes could be harmless as long as they are consistent across all
    # indices
    assert np.all(
        model.housing_xwalk == pd.Series([0, 1, 2, 3, 4], index=[0, 1, 2, 3, 4])
    )

    # make sure nbhd_for_choice is correct
    assert np.all(model.nbhd_for_choice == np.array([0, 0, 1, 1, 1]))

    # prop_white
    # First neighborhood is neighborhood 'a', should be 100% white b/c it contains households 0/1
    # second neighborhood is 'b', should be 33% white b/c it contains households 2/3/4
    assert np.allclose(model.endogenous_variables[:, 0], np.array([1, 1 / 3]))

    # mean income
    # First neighborhood is neighborhood 'a', should be 10 + 20 / 2 = 15
    # Second should be 30 + 40 + 50 / 3 = 40.
    assert np.allclose(
        model.endogenous_variables[:, 1], np.array([(10 + 20) / 2, (30 + 40 + 50) / 3])
    )

    # test that the interactions with the household variables are properly computed
    # construct the interaction term outside the model
    # first two choices are in 100% white nbhd, next three are 1/3
    intterm = (
        model.household_attributes.white.to_numpy()[model.full_hhidx]
        * np.array([1, 1, 1 / 3, 1 / 3, 1 / 3])[model.full_choiceidx]
    )

    assert model.alternatives_colnames[1] == "white:prop_white"

    # need to destandardize alternatives here, as they are roughly standardized to promote convergence, and
    # coefficients are automatically unstandardized after estimation.
    assert np.allclose((model.alternatives * model.alternatives_stds)[:, 1], intterm)

    # test that variables are updated correctly - this makes every household equally likely to choose each location
    faked_probabilities = np.full(model.alternatives.shape[0], 0.2)
    monkeypatch.setattr(model, "_probabilities", lambda: np.copy(faked_probabilities))

    # do the update
    model.initialize_or_update_endogenous_variables(initial=False)

    # every neighborhood should have 3 out of five households white because every neighborhood has same composition now
    assert np.allclose(model.endogenous_variables[:, 0], np.array([3 / 5, 3 / 5]))
    # and have a mean income of 30
    assert np.allclose(model.endogenous_variables[:, 1], np.array([30, 30]))


def test_endogenous_variables_with_weights(monkeypatch):
    model = TraSortingModel(
        housing_attributes=pd.DataFrame({"age": [10, 20, 30, 40, 50]}),
        household_attributes=pd.DataFrame({"white": [1, 1, 0, 0, 1]}),
        interactions=(("white", "prop_white"),),
        unequilibrated_hh_params=tuple(),
        unequilibrated_hsg_params=tuple(),
        second_stage_params=tuple(),
        # avoid price-income issues by making all prices less than all incomes - or maybe this should be tested
        # explicitly?
        price=pd.Series([5, 5, 5, 5, 5]),
        income=pd.Series([20, 20, 30, 40, 50]),
        choice=pd.Series([0, 1, 2, 3, 4]),
        weights=pd.Series([1, 1, 2, 1, 1]),
        unequilibrated_choice=pd.Series([0, 0, 0, 0, 0]),
        endogenous_variable_defs={
            "prop_white": lambda hh, inc, weights: np.average(
                hh.white, weights=weights
            ),
            "mean_income": lambda hh, inc, weights: np.average(inc, weights=weights),
        },
        neighborhoods=pd.Series(["a", "a", "b", "b", "b"]),
        fixed_price=0,
    )

    model.create_alternatives()

    model.initialize_or_update_endogenous_variables(initial=True)

    # as of Python 3.something, dict iteration order is the same as definition order. This test relies on this;
    # otherwise, we will not compare prop_white to expected prop_white. Can be fixed by actually reading the values in
    # endogenous_varnames
    assert model.endogenous_varnames == [
        "prop_white",
        "mean_income",
    ], "endogenous varnames incorrect or not in expected order"

    # as before, this could be harmless - but if our assumption in testing here was not met, later asserts would fail
    # in weird and wonderful ways
    assert np.all(
        model.nbhd_xwalk == pd.Series([0, 1], index=["a", "b"])
    ), "nbhd_xwalk not in expected order"

    # again, check that order is as expected - changes could be harmless as long as they are consistent across all
    # indices
    assert np.all(
        model.housing_xwalk == pd.Series([0, 1, 2, 3, 4], index=[0, 1, 2, 3, 4])
    )

    # make sure nbhd_for_choice is correct
    assert np.all(model.nbhd_for_choice == np.array([0, 0, 1, 1, 1]))

    # prop_white
    # First neighborhood is neighborhood 'a', should be 100% white b/c it contains households 0/1
    # second neighborhood is 'b', should be 25% white b/c it contains households 2/3/4, and hh 2 is weighted double
    assert np.allclose(model.endogenous_variables[:, 0], np.array([1, 1 / 4]))

    # mean income
    # First neighborhood is neighborhood 'a', should be 20 as both hhs are homogenous
    # Second should be 30 + 30 + 40 + 50 / 4 bc hh 4 is weighted twice
    assert np.allclose(
        model.endogenous_variables[:, 1], np.array([20, (30 + 30 + 40 + 50) / 4])
    )

    # test that the interactions with the household variables are properly computed
    # construct the interaction term outside the model
    # first two choices are in 100% white nbhd, next three are 1/3
    intterm = (
        model.household_attributes.white.to_numpy()[model.full_hhidx]
        * np.array([1, 1, 1 / 4, 1 / 4, 1 / 4])[model.full_choiceidx]
    )

    assert model.alternatives_colnames[1] == "white:prop_white"

    # need to destandardize alternatives here, as they are roughly standardized to promote convergence, and
    # coefficients are automatically unstandardized after estimation.
    assert np.allclose((model.alternatives * model.alternatives_stds)[:, 1], intterm)

    # test that variables are updated correctly - this makes every household equally likely to choose each location
    faked_probabilities = np.full(model.alternatives.shape[0], 0.2)
    monkeypatch.setattr(model, "_probabilities", lambda: np.copy(faked_probabilities))

    # do the update
    model.initialize_or_update_endogenous_variables(initial=False)

    # every neighborhood should have 3 out of 6 households white because every neighborhood has same composition now,
    # and one nonwhite hh is weighted double
    assert np.allclose(model.endogenous_variables[:, 0], np.array([3 / 6, 3 / 6]))
    # and have a mean income of 20, 20, 30, 30, 40, 50 b/c hh 2 is (30) is weighted double
    assert np.allclose(
        model.endogenous_variables[:, 1], np.full(2, np.mean([20, 20, 30, 30, 40, 50]))
    )
