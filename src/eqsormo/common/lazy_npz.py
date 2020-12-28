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
#         School of Geographical Sciences and Urban Planning
#         Arizona State University

# Lazy-load arrays from an NPZ file instead of loading everything at once

import numpy as np
from zipfile import ZipFile
import tempfile
import os


class LazyNPZ(object):
    """
    Lazy load Numpy arrays from an NPZ file.
    """

    def __init__(self, file_or_filename, allow_pickle=False):
        self.zipfile = ZipFile(file_or_filename)
        self.tempdir = tempfile.mkdtemp(prefix="npz")
        self.allow_pickle = allow_pickle

    def get_members(self):
        # strip off .npy
        return [i[:-4] for i in self.zipfile.namelist()]

    def get_member(self, item, mmap=False):
        "get a member, possibly loading it using mmap to save memory"
        if mmap:
            return self._get_member_mmap(item)
        else:
            return self._get_member(item)

    def _get_member(self, item):
        "get a member and return as bona-fide array"
        with self.zipfile.open(item + ".npy") as f:
            return np.load(f, allow_pickle=self.allow_pickle)

    def _get_member_mmap(self, item):
        "get a member and return as an mmapped-array"
        # extract the npy file
        npypath = self.zipfile.extract(item + ".npy", path=self.tempdir)
        return np.load(npypath, mmap_mode="r+", allow_pickle=self.allow_pickle)

    def __del__(self):
        self.zipfile.close()
        # can't delete temporary directory here, may be in use by mmapped objects
        # TODO figure out how to delete temp directory when all mmapped arrays are closed
