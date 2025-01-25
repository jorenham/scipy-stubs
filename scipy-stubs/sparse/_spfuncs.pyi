from typing import Literal as L, TypeAlias, TypeVar

import optype.numpy as onp
from ._base import _spbase
from ._typing import Floating

__all__ = ["count_blocks", "estimate_blocksize"]

_SizeT = TypeVar("_SizeT", bound=int)
_BlockSize: TypeAlias = tuple[_SizeT, _SizeT]

def estimate_blocksize(
    A: _spbase | onp.ToComplex2D,
    efficiency: float | Floating = 0.7,
) -> _BlockSize[L[1]] | _BlockSize[L[2]] | _BlockSize[L[3]] | _BlockSize[L[4]] | _BlockSize[L[6]]: ...

#
def count_blocks(A: _spbase | onp.ToComplex2D, blocksize: tuple[onp.ToJustInt, onp.ToJustInt]) -> int: ...
