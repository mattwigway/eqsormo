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

import dill
import pickle


class BaseSortingModel(object):
    """
    Represents a random utility based equilibrium sorting model.

    Based on the model described in Klaiber and Phaneuf (2010) "Valuing open space in a residential sorting model of the Twin Cities"
    https://doi.org/10.1016/j.jeem.2010.05.002 (no open access version available, unfortunately)
    """

    def to_pickle(self, fn):
        "Save a fit model to disk"
        if isinstance(fn, str):
            with open(fn, "wb") as out:
                dill.dump(self, out)
        else:
            dill.dump(self, fn)

    @classmethod
    def from_pickle(cls, fn):
        "Read a previously saved model"
        if isinstance(fn, str):
            with open(fn, "rb") as inf:
                model = dill.load(inf)
        else:
            model = dill.load(fn)

        if not isinstance(model, cls):
            raise ValueError("File does not contain a sorting model!")

        return model
