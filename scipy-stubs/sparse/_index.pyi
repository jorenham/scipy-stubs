from collections.abc import Sequence
from types import EllipsisType
from typing import Any, TypeAlias

import numpy as np
import optype as op
import optype.numpy as onp
from scipy._typing import Untyped
from ._base import _spbase

_Mask: TypeAlias = _spbase[np.bool_] | onp.CanArrayND[np.bool_] | Sequence[Sequence[bool]]
_Indexer: TypeAlias = op.CanIndex | onp.CanArrayND[np.bool_] | Sequence[bool] | onp.ArrayND[np.intp] | slice | EllipsisType
_Key: TypeAlias = tuple[_Indexer] | tuple[_Indexer, _Indexer] | Sequence[bool] | _Mask | EllipsisType

###

INT_TYPES: tuple[type[int], type[np.integer[Any]]] = ...

# TODO(jorenham): generic scalar type
class IndexMixin:
    def __getitem__(self, key: _Key, /) -> Untyped: ...
    def __setitem__(self, key: _Key, x: onp.ToComplex | onp.ToComplex1D | onp.ToComplex2D | _spbase, /) -> None: ...
