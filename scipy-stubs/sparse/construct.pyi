# ruff: noqa: ANN401
# This module is not meant for public use and will be removed in SciPy v2.0.0.

import numbers
from typing import Any
from typing_extensions import deprecated

from . import _bsr, _coo, _csc, _csr, _dia

__all__ = [
    "block_diag",
    "bmat",
    "bsr_matrix",
    "check_random_state",
    "coo_matrix",
    "csc_matrix",
    "csr_hstack",
    "csr_matrix",
    "dia_matrix",
    "diags",
    "eye",
    "get_index_dtype",
    "hstack",
    "identity",
    "isscalarlike",
    "issparse",
    "kron",
    "kronsum",
    "numbers",
    "rand",
    "random",
    "rng_integers",
    "spdiags",
    "upcast",
    "vstack",
]

@deprecated("will be removed in SciPy v2.0.0")
class coo_matrix(_coo.coo_matrix): ...

@deprecated("will be removed in SciPy v2.0.0")
class csc_matrix(_csc.csc_matrix): ...

@deprecated("will be removed in SciPy v2.0.0")
class csr_matrix(_csr.csr_matrix): ...

@deprecated("will be removed in SciPy v2.0.0")
class dia_matrix(_dia.dia_matrix): ...

@deprecated("will be removed in SciPy v2.0.0")
class bsr_matrix(_bsr.bsr_matrix): ...

@deprecated("will be removed in SciPy v2.0.0")
def csr_hstack(*args: object, **kwargs: object) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def issparse(x: object) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def check_random_state(seed: object) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def rng_integers(
    gen: object,
    low: object,
    high: object = ...,
    size: object = ...,
    dtype: object = ...,
    endpoint: object = ...,
) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def vstack(blocks: object, format: object = ..., dtype: object = ...) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def spdiags(data: object, diags: object, m: object = ..., n: object = ..., format: object = ...) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def random(
    m: object,
    n: object,
    density: object = ...,
    format: object = ...,
    dtype: object = ...,
    rng: object = ...,
    data_rvs: object = ...,
) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def rand(m: object, n: object, density: object = ..., format: object = ..., dtype: object = ..., rng: object = ...) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def kron(A: object, B: object, format: object = ...) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def kronsum(A: object, B: object, format: object = ...) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def identity(n: object, dtype: object = ..., format: object = ...) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def block_diag(mats: object, format: object = ..., dtype: object = ...) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def hstack(blocks: object, format: object = ..., dtype: object = ...) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def eye(m: object, n: object = ..., k: object = ..., dtype: object = ..., format: object = ...) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def diags(diagonals: object, offsets: object = ..., shape: object = ..., format: object = ..., dtype: object = ...) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def bmat(blocks: object, format: object = ..., dtype: object = ...) -> Any: ...

# sputils
@deprecated("will be removed in SciPy v2.0.0")
def get_index_dtype(arrays: object = ..., maxval: object = ..., check_contents: object = ...) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def isscalarlike(x: object) -> Any: ...
@deprecated("will be removed in SciPy v2.0.0")
def upcast(*args: object) -> Any: ...
