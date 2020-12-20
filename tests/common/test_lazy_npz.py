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

# Test the lazy NPZ module

import numpy as np
from eqsormo.common.lazy_npz import LazyNPZ
import tempfile
import os


def test_lazy_npz ():
    arr1 = np.arange(60)
    arr2 = np.repeat(['a', 'b', 'c'], 20)

    fh, tmpfn = tempfile.mkstemp(suffix='.npz')
    os.close(fh)
    np.savez(tmpfn, arr1=arr1, arr2=arr2)

    lazy = LazyNPZ(tmpfn)

    assert set(lazy.get_members()) == {'arr1', 'arr2'}
    
    arr1_read = lazy.get_member('arr1', mmap=False)
    assert len(arr1_read) == len(arr1)
    assert arr1_read.dtype == arr1.dtype
    assert np.all(arr1_read == arr1)

    arr2_read = lazy.get_member('arr2', mmap=False)
    assert len(arr2_read) == len(arr2)
    assert arr2_read.dtype == arr2.dtype
    assert np.all(arr2_read == arr2)

    # and now with mmap
    arr1_read = lazy.get_member('arr1', mmap=True)
    assert len(arr1_read) == len(arr1)
    assert arr1_read.dtype == arr1.dtype
    assert np.all(arr1_read == arr1)

    arr2_read = lazy.get_member('arr2', mmap=True)
    assert len(arr2_read) == len(arr2)
    assert arr2_read.dtype == arr2.dtype
    assert np.all(arr2_read == arr2)

    os.remove(tmpfn)

def test_lazy_npz_compressed ():
    arr1 = np.arange(60)
    arr2 = np.repeat(['a', 'b', 'c'], 20)

    fh, tmpfn = tempfile.mkstemp(suffix='.npz')
    os.close(fh)
    np.savez_compressed(tmpfn, arr1=arr1, arr2=arr2)

    lazy = LazyNPZ(tmpfn)

    assert set(lazy.get_members()) == {'arr1', 'arr2'}
    
    arr1_read = lazy.get_member('arr1', mmap=False)
    assert len(arr1_read) == len(arr1)
    assert arr1_read.dtype == arr1.dtype
    assert np.all(arr1_read == arr1)

    arr2_read = lazy.get_member('arr2', mmap=False)
    assert len(arr2_read) == len(arr2)
    assert arr2_read.dtype == arr2.dtype
    assert np.all(arr2_read == arr2)

    # and now with mmap
    arr1_read = lazy.get_member('arr1', mmap=True)
    assert len(arr1_read) == len(arr1)
    assert arr1_read.dtype == arr1.dtype
    assert np.all(arr1_read == arr1)

    arr2_read = lazy.get_member('arr2', mmap=True)
    assert len(arr2_read) == len(arr2)
    assert arr2_read.dtype == arr2.dtype
    assert np.all(arr2_read == arr2)

    os.remove(tmpfn)
