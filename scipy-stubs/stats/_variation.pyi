from collections.abc import Sequence
from typing import Any, Literal, overload
from typing_extensions import TypeVar

import numpy as np
import numpy.typing as npt
from numpy._typing import _ArrayLike, _ArrayLikeInt_co, _NestedSequence
from optype import CanIndex
from scipy._typing import AnyBool, AnyInt, NanPolicy

_SCT_fc = TypeVar("_SCT_fc", bound=np.inexact[Any])

###

# NOTE: This demonstrates the ridiculous complexity that's required to properly annotate this simple function with "array-likes".
# NOTE: Shape-typing hasn't even been included, as that would require even more overloads.

# array-like input with known floating- or complex-floating dtypes
@overload
def variation(
    a: _ArrayLike[_SCT_fc],
    axis: None,
    nan_policy: NanPolicy = "propagate",
    ddof: AnyInt = 0,
    *,
    keepdims: Literal[0, False] = False,
) -> _SCT_fc: ...
@overload
def variation(
    a: _ArrayLike[_SCT_fc],
    axis: CanIndex | None = 0,
    nan_policy: NanPolicy = "propagate",
    ddof: AnyInt = 0,
    *,
    keepdims: Literal[1, True],
) -> npt.NDArray[_SCT_fc]: ...
@overload
def variation(
    a: _ArrayLike[_SCT_fc],
    axis: CanIndex | None = 0,
    nan_policy: NanPolicy = "propagate",
    ddof: AnyInt = 0,
    *,
    keepdims: AnyBool = False,
) -> _SCT_fc | npt.NDArray[_SCT_fc]: ...

# sequences of `builtins.float`, that implicitly (and inevitably) also cover `builtins.int` and `builtins.bool`
@overload
def variation(
    a: Sequence[float],
    axis: CanIndex | None = 0,
    nan_policy: NanPolicy = "propagate",
    ddof: AnyInt = 0,
    *,
    keepdims: Literal[0, False] = False,
) -> np.float64: ...
@overload
def variation(
    a: _ArrayLikeInt_co | _NestedSequence[float],
    axis: None,
    nan_policy: NanPolicy = "propagate",
    ddof: AnyInt = 0,
    *,
    keepdims: Literal[0, False] = False,
) -> np.float64: ...
@overload
def variation(
    a: _ArrayLikeInt_co | _NestedSequence[float],
    axis: CanIndex | None = 0,
    nan_policy: NanPolicy = "propagate",
    ddof: AnyInt = 0,
    *,
    keepdims: Literal[1, True],
) -> npt.NDArray[np.float64]: ...
@overload
def variation(
    a: _ArrayLikeInt_co | _NestedSequence[float],
    axis: CanIndex | None = 0,
    nan_policy: NanPolicy = "propagate",
    ddof: AnyInt = 0,
    *,
    keepdims: AnyBool = False,
) -> np.float64 | npt.NDArray[np.float64]: ...

# sequences of `builtin.complex`, which behave as if `float <: complex` and therefore "overlaps" with the `builtins.float`
# overloads, hence the `complex128 | float64` returns
@overload
def variation(
    a: Sequence[complex],
    axis: CanIndex | None = 0,
    nan_policy: NanPolicy = "propagate",
    ddof: AnyInt = 0,
    *,
    keepdims: Literal[0, False] = False,
) -> np.complex128 | np.float64: ...
@overload
def variation(
    a: _NestedSequence[complex],
    axis: None,
    nan_policy: NanPolicy = "propagate",
    ddof: AnyInt = 0,
    *,
    keepdims: Literal[0, False] = False,
) -> np.complex128 | np.float64: ...
@overload
def variation(
    a: _NestedSequence[complex],
    axis: CanIndex | None = 0,
    nan_policy: NanPolicy = "propagate",
    ddof: AnyInt = 0,
    *,
    keepdims: Literal[1, True],
) -> npt.NDArray[np.complex128 | np.float64]: ...
@overload
def variation(
    a: _NestedSequence[complex],
    axis: CanIndex | None = 0,
    nan_policy: NanPolicy = "propagate",
    ddof: AnyInt = 0,
    *,
    keepdims: AnyBool = False,
) -> np.complex128 | np.float64 | npt.NDArray[np.complex128 | np.float64]: ...

# catch-all in case of broad gradual types
@overload
def variation(
    a: npt.ArrayLike,
    axis: CanIndex | None = 0,
    nan_policy: NanPolicy = "propagate",
    ddof: AnyInt = 0,
    *,
    keepdims: AnyBool = False,
) -> np.inexact[Any] | npt.NDArray[np.inexact[Any]]: ...
