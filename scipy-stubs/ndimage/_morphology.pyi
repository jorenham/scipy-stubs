from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt
from ._typing import _BoolValueIn, _FloatArrayIn, _IntArrayIn, _IntValueIn, _ScalarArrayIn, _ScalarArrayOut, _ScalarValueIn

__all__ = [
    "binary_closing",
    "binary_dilation",
    "binary_erosion",
    "binary_fill_holes",
    "binary_hit_or_miss",
    "binary_opening",
    "binary_propagation",
    "black_tophat",
    "distance_transform_bf",
    "distance_transform_cdt",
    "distance_transform_edt",
    "generate_binary_structure",
    "grey_closing",
    "grey_dilation",
    "grey_erosion",
    "grey_opening",
    "iterate_structure",
    "morphological_gradient",
    "morphological_laplace",
    "white_tophat",
]

_Mode: TypeAlias = Literal["reflect", "constant", "nearest", "mirror", "wrap"]
_MetricCDT: TypeAlias = Literal["chessboard", "taxicab"]
_MetricBF: TypeAlias = Literal["euclidean", _MetricCDT]
_BorderValue: TypeAlias = _IntValueIn | np.bool_

_BoolArrayOut: TypeAlias = npt.NDArray[np.bool_]
_Origin: TypeAlias = int | tuple[int, ...]

def iterate_structure(structure: _IntArrayIn, iterations: _IntValueIn, origin: _Origin | None = None) -> _BoolArrayOut: ...
def generate_binary_structure(rank: int, connectivity: int) -> _BoolArrayOut: ...
def binary_erosion(
    input: _ScalarArrayIn,
    structure: _IntArrayIn | None = None,
    iterations: _IntValueIn = 1,
    mask: _IntArrayIn | None = None,
    output: _BoolArrayOut | type[bool | np.bool_] | None = None,
    border_value: _BorderValue = 0,
    origin: _Origin = 0,
    brute_force: _BoolValueIn = False,
) -> _BoolArrayOut: ...
def binary_dilation(
    input: _ScalarArrayIn,
    structure: _IntArrayIn | None = None,
    iterations: _IntValueIn = 1,
    mask: _IntArrayIn | None = None,
    output: _BoolArrayOut | type[bool | np.bool_] | None = None,
    border_value: _BorderValue = 0,
    origin: _Origin = 0,
    brute_force: _BoolValueIn = False,
) -> _BoolArrayOut: ...
def binary_opening(
    input: _ScalarArrayIn,
    structure: _IntArrayIn | None = None,
    iterations: _IntValueIn = 1,
    output: _BoolArrayOut | type[bool | np.bool_] | None = None,
    origin: _Origin = 0,
    mask: _IntArrayIn | None = None,
    border_value: _BorderValue = 0,
    brute_force: _BoolValueIn = False,
) -> _BoolArrayOut: ...
def binary_closing(
    input: _ScalarArrayIn,
    structure: _IntArrayIn | None = None,
    iterations: _IntValueIn = 1,
    output: _BoolArrayOut | type[bool | np.bool_] | None = None,
    origin: _Origin = 0,
    mask: _IntArrayIn | None = None,
    border_value: _BorderValue = 0,
    brute_force: _BoolValueIn = False,
) -> _BoolArrayOut: ...
def binary_hit_or_miss(
    input: _ScalarArrayIn,
    structure1: _IntArrayIn | None = None,
    structure2: _IntArrayIn | None = None,
    output: _BoolArrayOut | type[bool | np.bool_] | None = None,
    origin1: _Origin = 0,
    origin2: _Origin | None = None,
) -> _BoolArrayOut: ...
def binary_propagation(
    input: _ScalarArrayIn,
    structure: _IntArrayIn | None = None,
    mask: _IntArrayIn | None = None,
    output: _BoolArrayOut | type[bool | np.bool_] | None = None,
    border_value: _BorderValue = 0,
    origin: _Origin = 0,
) -> _BoolArrayOut: ...
def binary_fill_holes(
    input: _ScalarArrayIn,
    structure: _IntArrayIn | None = None,
    output: _BoolArrayOut | None = None,
    origin: _Origin = 0,
) -> _BoolArrayOut: ...
def grey_erosion(
    input: _ScalarArrayIn,
    size: tuple[int, ...] | None = None,
    footprint: npt.ArrayLike | None = None,
    structure: _IntArrayIn | None = None,
    output: _ScalarArrayOut | None = None,
    mode: _Mode = "reflect",
    cval: _ScalarValueIn = 0.0,
    origin: _Origin = 0,
) -> _ScalarArrayOut: ...
def grey_dilation(
    input: _ScalarArrayIn,
    size: tuple[int, ...] | None = None,
    footprint: npt.ArrayLike | None = None,
    structure: _IntArrayIn | None = None,
    output: _ScalarArrayOut | None = None,
    mode: _Mode = "reflect",
    cval: _ScalarValueIn = 0.0,
    origin: _Origin = 0,
) -> _ScalarArrayOut: ...
def grey_opening(
    input: _ScalarArrayIn,
    size: tuple[int, ...] | None = None,
    footprint: npt.ArrayLike | None = None,
    structure: _IntArrayIn | None = None,
    output: _ScalarArrayOut | None = None,
    mode: _Mode = "reflect",
    cval: _ScalarValueIn = 0.0,
    origin: _Origin = 0,
) -> _ScalarArrayOut: ...
def grey_closing(
    input: _ScalarArrayIn,
    size: tuple[int, ...] | None = None,
    footprint: npt.ArrayLike | None = None,
    structure: _IntArrayIn | None = None,
    output: _ScalarArrayOut | None = None,
    mode: _Mode = "reflect",
    cval: _ScalarValueIn = 0.0,
    origin: _Origin = 0,
) -> _ScalarArrayOut: ...
def morphological_gradient(
    input: _ScalarArrayIn,
    size: tuple[int, ...] | None = None,
    footprint: npt.ArrayLike | None = None,
    structure: _IntArrayIn | None = None,
    output: _ScalarArrayOut | None = None,
    mode: _Mode = "reflect",
    cval: _ScalarValueIn = 0.0,
    origin: _Origin = 0,
) -> _ScalarArrayOut: ...
def morphological_laplace(
    input: _ScalarArrayIn,
    size: tuple[int, ...] | None = None,
    footprint: npt.ArrayLike | None = None,
    structure: _IntArrayIn | None = None,
    output: _ScalarArrayOut | None = None,
    mode: _Mode = "reflect",
    cval: _ScalarValueIn = 0.0,
    origin: _Origin = 0,
) -> _ScalarArrayOut: ...
def white_tophat(
    input: _ScalarArrayIn,
    size: tuple[int, ...] | None = None,
    footprint: npt.ArrayLike | None = None,
    structure: _IntArrayIn | None = None,
    output: _ScalarArrayOut | None = None,
    mode: _Mode = "reflect",
    cval: _ScalarValueIn = 0.0,
    origin: _Origin = 0,
) -> _ScalarArrayOut: ...
def black_tophat(
    input: _ScalarArrayIn,
    size: tuple[int, ...] | None = None,
    footprint: npt.ArrayLike | None = None,
    structure: _IntArrayIn | None = None,
    output: _ScalarArrayOut | None = None,
    mode: _Mode = "reflect",
    cval: _ScalarValueIn = 0.0,
    origin: _ScalarValueIn = 0,
) -> _ScalarArrayOut: ...
def distance_transform_bf(
    input: _ScalarArrayIn,
    metric: _MetricBF = "euclidean",
    sampling: _FloatArrayIn | None = None,
    return_distances: _BoolValueIn = True,
    return_indices: _BoolValueIn = False,
    distances: npt.NDArray[np.float64 | np.uint32] | None = None,
    indices: npt.NDArray[np.int32] | None = None,
) -> _ScalarArrayOut | npt.NDArray[np.int32] | tuple[_ScalarArrayOut, npt.NDArray[np.int32]]: ...
def distance_transform_cdt(
    input: _ScalarArrayIn,
    metric: _MetricCDT | npt.ArrayLike = "chessboard",
    return_distances: _BoolValueIn = True,
    return_indices: _BoolValueIn = False,
    distances: npt.NDArray[np.int32] | None = None,
    indices: npt.NDArray[np.int32] | None = None,
) -> npt.NDArray[np.int32] | tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]: ...
def distance_transform_edt(
    input: _ScalarArrayIn,
    sampling: npt.ArrayLike | None = None,
    return_distances: _BoolValueIn = True,
    return_indices: _BoolValueIn = False,
    distances: npt.NDArray[np.float64] | None = None,
    indices: npt.NDArray[np.int32] | None = None,
) -> npt.NDArray[np.float64] | npt.NDArray[np.int32] | tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]: ...
