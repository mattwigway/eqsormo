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

import logging

# from .sorting_model import SortingModel
from .tra import TraSortingModel
from . import tra

version = "0.5.1"

rootLogger = None


def enable_logging():
    """
    Enable logging for the EqSorMo package.

    EqSorMo uses the Python logging package. If you are managing loggers at a higher level in a large project, you may not want to run this
    function. Otherwise, running this function will create a root logger and enable logging to the console.
    """
    global rootLogger

    if rootLogger is None:
        rootLogger = logging.getLogger()
        rootLogger.setLevel(logging.INFO)
        rootLogger.addHandler(logging.StreamHandler())

        rootLogger.info(f"Eqsormo version {version}")
    else:
        rootLogger.info("Logging already enabled.")
