from collections.abc import Sequence
from typing import Any, Literal, TypeAlias, overload
from typing_extensions import TypeVar

import numpy as np
import numpy.typing as npt
import optype.numpy as onpt

__all__ = ["kmeans", "kmeans2", "vq", "whiten"]

_SCT_fc = TypeVar("_SCT_fc", bound=np.inexact[Any])
_ArrayLike_1d_fc: TypeAlias = onpt.AnyNumberArray | Sequence[complex | np.number[Any]]
_ArrayLike_2d_fc: TypeAlias = onpt.AnyNumberArray | Sequence[Sequence[complex | np.number[Any]]]

class ClusterError(Exception): ...

@overload
def whiten(obs: npt.NDArray[_SCT_fc], check_finite: bool = True) -> onpt.Array[tuple[int, int], _SCT_fc]: ...
@overload
def whiten(obs: _ArrayLike_2d_fc, check_finite: bool = True) -> onpt.Array[tuple[int, int], np.inexact[Any]]: ...
def vq(
    obs: _ArrayLike_2d_fc,
    code_book: _ArrayLike_2d_fc,
    check_finite: bool = True,
) -> tuple[onpt.Array[tuple[int], np.int32 | np.intp], onpt.Array[tuple[int], _SCT_fc]]: ...
def py_vq(
    obs: _ArrayLike_2d_fc,
    code_book: _ArrayLike_2d_fc,
    check_finite: bool = True,
) -> tuple[onpt.Array[tuple[int], np.intp], onpt.Array[tuple[int], _SCT_fc]]: ...
def kmeans(
    obs: _ArrayLike_2d_fc,
    k_or_guess: npt.ArrayLike,
    iter: int = 20,
    thresh: float = 1e-05,
    check_finite: bool = True,
    *,
    seed: int | np.random.Generator | np.random.RandomState | None = None,
) -> tuple[onpt.Array[tuple[int, int], np.inexact[Any]], float]: ...
def kmeans2(
    data: _ArrayLike_1d_fc | _ArrayLike_2d_fc,
    k: npt.ArrayLike,
    iter: int = 10,
    thresh: float = 1e-05,
    minit: Literal["random", "points", "++", "matrix"] = "random",
    missing: Literal["warn", "raise"] = "warn",
    check_finite: bool = True,
    *,
    seed: int | np.random.Generator | np.random.RandomState | None = None,
) -> tuple[onpt.Array[tuple[int, int], np.inexact[Any]], onpt.Array[tuple[int], np.int32]]: ...
