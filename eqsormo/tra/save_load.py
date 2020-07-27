MODEL_VERSION = 0

import numpy as np
import dill

# This is very hacky, but what we are doing here is saving two files, on with numpy arrays in it, and one with
# everything else. This will hopefully work until we can come up with something better...

FIELDS = [
 'choice',
 'creation_time',
 'est_first_stage_ses',
 'first_stage_ascs',
 'first_stage_fit',
 'first_stage_uneq_ascs',
 'full_choiceidx',
 'full_chosen',
 'full_hh_hsgidx',
 'full_hhidx',
 'full_hsgchosen',
 'full_uneqchoiceidx',
 'full_uneqchosen',
 'full_utility',
 'hh_hsg_choice',
 'hh_unequilibrated_choice',
 'hh_xwalk',
 'household_attributes',
 'household_housing_attributes',
 'housing_attributes',
 'housing_xwalk',
 'income',
 'interactions',
 'max_chunk_bytes',
 'max_rent_to_income',
 'method',
 'orig_price',
 'price',
 'price_income_starting_values',
 'price_income_transformation',
 'sample_alternatives',
 'second_stage_fit',
 'second_stage_params',
 'seed',
 'type_shock',
 'unequilibrated_choice',
 'unequilibrated_choice_xwalk',
 'unequilibrated_hh_params',
 'unequilibrated_hsg_params',
 'weighted_supply',
 'weights'
 ]

def save (basefile, model):
    pickle_fields = {
        'MODEL_VERSION': MODEL_VERSION
    }
    numpy_fields = {
        'MODEL_VERSION': MODEL_VERSION
    }

    for field in FIELDS:
        val = getattr(model, field)
        if isinstance(val, np.ndarray):
            numpy_fields[field] = val
        else:
            pickle_fields[field] = val

    with open(f'{basefile}.pickle', 'wb') as out:
        dill.dump(pickle_fields, out)

    np.savez_compressed(f'{basefile}.npz', **numpy_fields)
