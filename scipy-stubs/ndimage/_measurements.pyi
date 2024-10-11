from collections.abc import Callable, Sequence
from typing import Any, Protocol, TypeAlias, overload, type_check_only
from typing_extensions import TypeVar

import numpy as np
import numpy.typing as npt
from numpy._typing import _ArrayLike, _DTypeLike, _NestedSequence
from ._typing import _FloatValueIn, _IntArrayIn, _IntValueIn, _ScalarArrayIn, _ScalarValueIn, _ScalarValueOut

__all__ = [
    "center_of_mass",
    "extrema",
    "find_objects",
    "histogram",
    "label",
    "labeled_comprehension",
    "maximum",
    "maximum_position",
    "mean",
    "median",
    "minimum",
    "minimum_position",
    "standard_deviation",
    "sum",
    "sum_labels",
    "value_indices",
    "variance",
    "watershed_ift",
]

_SCT = TypeVar("_SCT", bound=_ScalarValueOut, default=_ScalarValueOut)
_ISCT = TypeVar("_ISCT", bound=np.inexact[Any], default=np.inexact[Any])

__Func1: TypeAlias = Callable[[_ScalarArrayIn], _ScalarValueIn]
__Func2: TypeAlias = Callable[[_ScalarArrayIn, _ScalarArrayIn], _ScalarValueIn]
_ComprehensionFunc: TypeAlias = __Func1 | __Func2

_Val0D: TypeAlias = _SCT
_ValND: TypeAlias = npt.NDArray[_SCT]

_IVal0D: TypeAlias = _ISCT
_IValND: TypeAlias = npt.NDArray[_ISCT]
_IVal_D: TypeAlias = _ISCT | npt.NDArray[_ISCT]

_Idx0D: TypeAlias = tuple[np.intp, ...]
_IdxND: TypeAlias = list[_Idx0D]

_Extrema0D: TypeAlias = tuple[_Val0D[_SCT], _Val0D[_SCT], _Idx0D, _Idx0D]
_ExtremaND: TypeAlias = tuple[_ValND[_SCT], _ValND[_SCT], _IdxND, _IdxND]

_Coord0D: TypeAlias = tuple[np.float64, ...]
_Coord1D: TypeAlias = list[_Coord0D]
_CoordND: TypeAlias = list[tuple[npt.NDArray[np.float64], ...]]

_IntArrayND: TypeAlias = npt.NDArray[np.integer[Any]]

#
def label(
    input: _ScalarArrayIn,
    structure: _ScalarArrayIn | None = None,
    output: npt.NDArray[np.int32 | np.intp] | None = None,
) -> int | tuple[npt.NDArray[np.int32 | np.intp], int]: ...

#
def find_objects(input: _IntArrayIn, max_label: int = 0) -> list[tuple[slice, ...]]: ...

#
def value_indices(arr: _IntArrayIn, *, ignore_value: int | None = None) -> dict[np.intp, tuple[npt.NDArray[np.intp], ...]]: ...

#
@overload
def labeled_comprehension(
    input: _ScalarArrayIn,
    labels: _ScalarArrayIn | None,
    index: _IntValueIn | _IntArrayIn | None,
    func: _ComprehensionFunc,
    out_dtype: _DTypeLike[_SCT],
    default: _FloatValueIn,
    pass_positions: bool = False,
) -> npt.NDArray[_SCT]: ...
@overload
def labeled_comprehension(
    input: _ScalarArrayIn,
    labels: _ScalarArrayIn | None,
    index: _IntValueIn | _IntArrayIn | None,
    func: _ComprehensionFunc,
    out_dtype: type[int],
    default: _IntValueIn,
    pass_positions: bool = False,
) -> npt.NDArray[np.intp]: ...
@overload
def labeled_comprehension(
    input: _ScalarArrayIn,
    labels: _ScalarArrayIn | None,
    index: _IntValueIn | _IntArrayIn | None,
    func: _ComprehensionFunc,
    out_dtype: type[float],
    default: _FloatValueIn,
    pass_positions: bool = False,
) -> npt.NDArray[np.float64 | np.intp]: ...
@overload
def labeled_comprehension(
    input: _ScalarArrayIn,
    labels: _ScalarArrayIn | None,
    index: _IntValueIn | _IntArrayIn | None,
    func: _ComprehensionFunc,
    out_dtype: type[complex],
    default: _ScalarValueIn,
    pass_positions: bool = False,
) -> npt.NDArray[np.complex128 | np.float64 | np.intp]: ...

#
@type_check_only
class _DefStatistic(Protocol):
    @overload
    def __call__(self, /, input: _ArrayLike[_ISCT], labels: _IntArrayIn | None = None, index: None = None) -> _IVal0D[_ISCT]: ...
    @overload
    def __call__(self, /, input: _IntArrayIn, labels: _IntArrayIn | None = None, index: None = None) -> _IVal0D[np.float64]: ...
    @overload
    def __call__(self, /, input: _ScalarArrayIn, labels: _IntArrayIn | None = None, index: None = None) -> _IVal0D: ...
    @overload
    def __call__(self, /, input: _ArrayLike[_ISCT], labels: _IntArrayIn, index: _IntArrayND) -> _IValND[_ISCT]: ...
    @overload
    def __call__(self, /, input: _IntArrayIn, labels: _IntArrayIn, index: _IntArrayND) -> _IValND[np.float64]: ...
    @overload
    def __call__(self, /, input: _ScalarArrayIn, labels: _IntArrayIn, index: _IntArrayND) -> _IValND: ...
    @overload
    def __call__(self, /, input: _ArrayLike[_ISCT], labels: _IntArrayIn, index: _IntArrayIn) -> _IVal_D[_ISCT]: ...
    @overload
    def __call__(self, /, input: _ScalarArrayIn, labels: _IntArrayIn, index: _IntArrayIn) -> _IVal_D: ...

sum: _DefStatistic
sum_labels: _DefStatistic
mean: _DefStatistic
variance: _DefStatistic
standard_deviation: _DefStatistic
median: _DefStatistic

#
@type_check_only
class _DefExtreme(Protocol):
    @overload
    def __call__(self, /, input: _ArrayLike[_SCT], labels: _IntArrayIn | None = None, index: None = None) -> _Val0D[_SCT]: ...
    @overload
    def __call__(self, /, input: _ScalarArrayIn, labels: _IntArrayIn | None = None, index: None = None) -> _Val0D: ...
    @overload
    def __call__(self, /, input: _ArrayLike[_SCT], labels: _IntArrayIn, index: _IntArrayND) -> _ValND[_SCT]: ...
    @overload
    def __call__(self, /, input: _ScalarArrayIn, labels: _IntArrayIn, index: _IntArrayND) -> _ValND: ...
    @overload
    def __call__(self, /, input: _ArrayLike[_SCT], labels: _IntArrayIn, index: _IntArrayIn) -> _Val0D[_SCT] | _ValND[_SCT]: ...
    @overload
    def __call__(self, /, input: _ScalarArrayIn, labels: _IntArrayIn, index: _IntArrayIn) -> _Val0D | _ValND: ...

minimum: _DefExtreme
maximum: _DefExtreme

#
@overload
def extrema(input: _ArrayLike[_SCT], labels: _IntArrayIn | None = None, index: _IntValueIn | None = None) -> _Extrema0D[_SCT]: ...
@overload
def extrema(input: _ScalarArrayIn, labels: _IntArrayIn | None = None, index: _IntValueIn | None = None) -> _Extrema0D: ...
@overload
def extrema(input: _ArrayLike[_SCT], labels: _IntArrayIn, index: _IntArrayND) -> _ExtremaND[_SCT]: ...
@overload
def extrema(input: _ScalarArrayIn, labels: _IntArrayIn, index: _IntArrayND) -> _ExtremaND: ...
@overload
def extrema(input: _ArrayLike[_SCT], labels: _IntArrayIn, index: _IntArrayIn) -> _Extrema0D[_SCT] | _ExtremaND[_SCT]: ...
@overload
def extrema(input: _ScalarArrayIn, labels: _IntArrayIn, index: _IntArrayIn) -> _Extrema0D | _ExtremaND: ...

#
@type_check_only
class _DefArgExtreme(Protocol):
    @overload
    def __call__(self, /, input: _ScalarArrayIn, labels: _IntArrayIn | None = None, index: None = None) -> _Idx0D: ...
    @overload
    def __call__(self, /, input: _ScalarArrayIn, labels: _IntArrayIn, index: _IntArrayND) -> _IdxND: ...
    @overload
    def __call__(self, /, input: _ScalarArrayIn, labels: _IntArrayIn, index: _IntArrayIn) -> _Idx0D | _IdxND: ...

minimum_position: _DefArgExtreme
maximum_position: _DefArgExtreme

#
@overload
def center_of_mass(input: _ScalarArrayIn, labels: _IntArrayIn | None = None, index: _IntValueIn | None = None) -> _Coord0D: ...
@overload
def center_of_mass(input: _ScalarArrayIn, labels: _IntArrayIn, index: Sequence[_IntValueIn]) -> _Coord1D: ...
@overload
def center_of_mass(input: _ScalarArrayIn, labels: _IntArrayIn, index: Sequence[Sequence[_IntArrayIn]]) -> _CoordND: ...
@overload
def center_of_mass(input: _ScalarArrayIn, labels: _IntArrayIn, index: _IntArrayIn) -> _Coord0D | _Coord1D | _CoordND: ...

#
@overload
def histogram(
    input: _ScalarArrayIn,
    min: _IntValueIn,
    max: _IntValueIn,
    bins: _IntValueIn,
    labels: _IntArrayIn | None = None,
    index: _IntValueIn | None = None,
) -> npt.NDArray[np.intp]: ...
@overload
def histogram(
    input: _ScalarArrayIn,
    min: _IntValueIn,
    max: _IntValueIn,
    bins: _IntValueIn,
    labels: _IntArrayIn,
    index: _IntArrayND,
) -> npt.NDArray[np.object_]: ...
@overload
def histogram(
    input: _ScalarArrayIn,
    min: _IntValueIn,
    max: _IntValueIn,
    bins: _IntValueIn,
    labels: _IntArrayIn,
    index: _IntArrayIn,
) -> npt.NDArray[np.intp | np.object_]: ...

#
def watershed_ift(
    input: _ArrayLike[np.uint8 | np.uint16],
    markers: _ArrayLike[np.signedinteger[Any]] | _NestedSequence[int],
    structure: _IntArrayIn | None = None,
    output: npt.NDArray[np.signedinteger[Any]] | None = None,
) -> npt.NDArray[np.signedinteger[Any]]: ...
