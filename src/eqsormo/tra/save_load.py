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

# Author: Matthew Wigginton Conway <matt@indicatrix.org>
#         School of Geographical Sciences and Urban Planning,
#         Arizona State University

import numpy as np
import dill
import logging
import uuid
from eqsormo.common.lazy_npz import LazyNPZ

MODEL_VERSION = 0

LOG = logging.getLogger(__name__)

# This is very hacky, but what we are doing here is saving two files, on with numpy arrays in it, and one with
# everything else. This will hopefully work until we can come up with something better...

FIELDS = [
    "alternatives_colnames",
    "alternatives_stds",
    "choice",
    "creation_time",
    "endogenous_variable_defs",
    "endogenous_variables",
    "endogenous_varnames",
    "est_first_stage_ses",
    "first_stage_ascs",
    "first_stage_fit",
    "first_stage_uneq_ascs",
    "full_choiceidx",
    "full_chosen",
    "full_hh_hsgidx",
    "full_hhidx",
    "full_hsgchosen",
    "full_uneqchoiceidx",
    "full_uneqchosen",
    "hh_hsg_choice",
    "hh_unequilibrated_choice",
    "hh_xwalk",
    "household_attributes",
    "household_housing_attributes",
    "housing_attributes",
    "housing_xwalk",
    "income",
    "interactions",
    "max_chunk_bytes",
    "max_rent_to_income",
    "method",
    "minimize_options",
    "nbhd_xwalk",
    "nbhd_for_choice",
    "neighborhoods",
    "orig_endogenous_variables",
    "orig_price",
    "price",
    "price_income_starting_values",
    "price_income_transformation",
    "sample_alternatives",
    "second_stage_fit",
    "second_stage_params",
    "seed",
    "type_shock",
    "unequilibrated_choice",
    "unequilibrated_choice_xwalk",
    "unequilibrated_hh_params",
    "unequilibrated_hsg_params",
    "weighted_supply",
    "weights",
]

# fields that should be mmapped to save memory
# these are large fields that we only use part of
MMAP_FIELDS = [
    "full_choiceidx",
    "full_chosen",
    "full_hh_hsgidx",
    "full_hhidx",
    "full_hsgchosen",
    "full_uneqchoiceidx",
    "full_uneqchosen",
    "hh_hsg_choice",
    "hh_unequilibrated_choice",
    "hh_xwalk",
    "household_housing_attributes",
]


def save(basefile, model):
    model_uuid = uuid.uuid4().hex

    pickle_fields = {"MODEL_VERSION": MODEL_VERSION, "MODEL_ID": model_uuid}
    numpy_fields = {
        "MODEL_VERSION": np.array([MODEL_VERSION]),
        "MODEL_ID": np.array([model_uuid]),
    }

    for field in FIELDS:
        val = getattr(model, field)
        if isinstance(val, np.ndarray):
            numpy_fields[field] = val
        else:
            pickle_fields[field] = val

    with open(f"{basefile}.pickle", "wb") as out:
        dill.dump(pickle_fields, out)

    # TODO any way to set allow_pickle=False here?
    np.savez_compressed(f"{basefile}.npz", **numpy_fields)


def load(basefile):
    # do here to prevent circular import
    from .tra_sorting_model import TraSortingModel

    with open(f"{basefile}.pickle", "rb") as inp:
        pkl = dill.load(inp)

    npz = LazyNPZ(f"{basefile}.npz", allow_pickle=False)
    npz_contents = npz.get_members()

    if (
        pkl["MODEL_VERSION"] != MODEL_VERSION
        or npz.get_member("MODEL_VERSION")[0] != MODEL_VERSION
    ):
        # TODO npz model version not saved correctly so not checked
        raise ValueError(
            f'Model version {pkl["MODEL_VERSION"]} does not match model version {MODEL_VERSION}'
        )

    if pkl["MODEL_ID"] != npz.get_member("MODEL_ID")[0]:
        raise ValueError("Model UUIDs do not match")

    def get(field):
        if field in pkl:
            return pkl[field]
        elif field in npz_contents:
            return npz.get_member(field, mmap=field in MMAP_FIELDS)
        else:
            LOG.warn(f"Field {field} not found in stored model, assuming it was none")
            return None

    model = TraSortingModel(
        housing_attributes=get("housing_attributes"),
        household_attributes=get("household_attributes"),
        interactions=get("interactions"),
        unequilibrated_hh_params=get("unequilibrated_hh_params"),
        unequilibrated_hsg_params=get("unequilibrated_hsg_params"),
        second_stage_params=get("second_stage_params"),
        price=get("price"),
        income=get("income"),
        choice=get("choice"),
        unequilibrated_choice=get("unequilibrated_choice"),
        price_income_transformation=get("price_income_transformation"),
        price_income_starting_values=get("price_income_starting_values"),
        sample_alternatives=get("sample_alternatives"),
        method=get("method"),
        max_rent_to_income=get("max_rent_to_income"),
        household_housing_attributes=get("household_housing_attributes"),
        weights=get("weights"),
        minimize_options=get("minimize_options"),
        neighborhoods=get("neighborhoods"),
        endogenous_variable_defs=get("endogenous_variable_defs"),
        max_chunk_bytes=get("max_chunk_bytes"),
        est_first_stage_ses=get("est_first_stage_ses"),
        seed=get("seed"),
    )

    for field in FIELDS:
        setattr(model, field, get(field))

    return model
