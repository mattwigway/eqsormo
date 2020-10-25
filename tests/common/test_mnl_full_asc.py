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

"""
Tests for the multinomial logit code
"""

import pytest
import numpy as np
import pandas as pd
import os
from itertools import product
import statsmodels.tools.numdiff
from eqsormo.common import MNLFullASC


@pytest.fixture
def vehchoice():
    """
    Dataset for vehicle choice model, used to test MNLFullASC. Extracted from IPUMS.
    """

    return pd.read_parquet(
        os.path.join(os.path.dirname(__file__), "data", "veh_choice.parquet.gz")
    )


@pytest.fixture
def vehchoice_alts():
    """
    Dataset for vehicle choice model, used to test MNLFullASC. Extracted from IPUMS. Formatted for use in MNLFullASC
    """

    est_data = pd.read_parquet(
        os.path.join(os.path.dirname(__file__), "data", "veh_choice.parquet.gz")
    )

    def get_data_for_choice(cars, sfh):
        chcdata = pd.DataFrame(index=est_data.index)
        chcdata["houseid"] = est_data.houseid
        chcdata["alt_cars"] = cars
        chcdata["alt_sfh"] = sfh
        if cars != 0:
            chcdata[f"college_{cars}_cars"] = est_data.college
            chcdata[f"inc_{cars}_cars"] = est_data.hhincome
        if sfh != 0:
            chcdata["college_sfh"] = est_data.college
            chcdata["inc_sfh"] = est_data.hhincome

        chcdata["sfh_act"] = est_data.sfh
        chcdata["cars_act"] = est_data.nveh
        return chcdata

    alts = (
        pd.concat(
            [
                get_data_for_choice(cars, sfh)
                for cars, sfh in product([0, 1, 2, 3], [0, 1])
            ],
            ignore_index=True,
        )
        .fillna(0)
        .sort_values(["houseid", "alt_sfh", "alt_cars"])
        .reset_index(drop=True)
    )

    houseidx = alts.houseid.values
    car_choiceidx = alts.alt_cars.values
    sfh_choiceidx = alts.alt_sfh.values
    car_chosen = (alts.cars_act == alts.alt_cars).values
    sfh_chosen = (alts.sfh_act == alts.alt_sfh).values
    chosen = car_chosen & sfh_chosen
    alt_data = alts.drop(
        columns=["houseid", "alt_cars", "alt_sfh", "sfh_act", "cars_act"]
    )
    alt_data_colnames = alt_data.columns
    alt_data = alt_data.values
    alt_data_stds = np.std(alt_data, axis=0)
    alt_data /= alt_data_stds

    return {
        "data": alt_data,
        "houseidx": houseidx,
        "car_choiceidx": car_choiceidx,
        "sfh_choiceidx": sfh_choiceidx,
        "car_chosen": car_chosen,
        "sfh_chosen": sfh_chosen,
        "chosen": chosen,
        "colnames": alt_data_colnames,
        "stds": alt_data_stds,
    }


def test_hessian(vehchoice, vehchoice_alts, monkeypatch):
    """
    We have a custom, faster Hessian estimation procedure that takes advantage of some constant components of utility.
    Make sure it produces the same results as the statsmodels numdiff tools.
    """
    mfa = MNLFullASC(
        vehchoice_alts["data"],
        supply=(np.bincount(vehchoice.nveh), np.bincount(vehchoice.sfh)),
        hhidx=vehchoice_alts["houseidx"],
        choiceidx=(vehchoice_alts["car_choiceidx"], vehchoice_alts["sfh_choiceidx"]),
        chosen=vehchoice_alts["chosen"],
        starting_values=np.zeros(vehchoice_alts["data"].shape[1]),
        param_names=vehchoice_alts["colnames"],
    )

    mfa.fit()

    hess_fast = mfa.linear_hessian(mfa.params.to_numpy())
    hess_fast_inv = np.linalg.inv(hess_fast)

    # the fast Hessian calculation above does not adjust ASCs with each change to the parameters. Monkeypatch the model
    # not to either
    monkeypatch.setattr(mfa, "compute_ascs", lambda u: mfa.ascs)

    # approx_hess3 uses the same algorithm we are using
    hess_sm = statsmodels.tools.numdiff.approx_hess3(
        mfa.params.to_numpy(), mfa.negative_log_likelihood
    )
    hess_sm_inv = np.linalg.inv(hess_sm)

    # compare the inverse Hessians, rather than the Hessians, as there appears to be significant roundoff error
    # in the Hessians
    # TODO roundoff is a lot here?
    assert np.allclose(
        hess_fast_inv, hess_sm_inv, atol=1e-5, rtol=1e-5
    ), "Inverse Hessian does not match statsmodels!"
