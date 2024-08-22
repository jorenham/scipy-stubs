from array_api_compat import (
    is_cupy_array as is_cupy_array,
    is_dask_array as is_dask_array,
    is_jax_array as is_jax_array,
    is_numpy_array as is_numpy_array,
    is_pydata_sparse_array as is_pydata_sparse_array,
    is_torch_array as is_torch_array,
)

from ._helpers import all_libraries as all_libraries, import_ as import_, wrapped_libraries as wrapped_libraries
from scipy._typing import Untyped

is_functions: Untyped

def test_is_xp_array(library, func): ...
def test_device(library): ...
def test_to_device_host(library): ...
def test_asarray_cross_library(source_library, target_library, request): ...
def test_asarray_copy(library) -> Untyped: ...
