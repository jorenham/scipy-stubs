# type-check-only typing utilities for internal usage
from typing import Any, Literal, TypeAlias, TypedDict, type_check_only
from typing_extensions import NotRequired

import numpy as np
import numpy.typing as npt
import optype.numpy as onp
from numpy._typing import _ArrayLikeFloat_co

__all__ = "ODEInfoDict", "QuadInfoDict", "QuadOpts", "QuadWeights"

_IntLike: TypeAlias = int | np.integer[Any]
_FloatLike: TypeAlias = float | np.floating[Any]

QuadWeights: TypeAlias = Literal["cos", "sin", "alg", "alg-loga", "alg-logb", "alg-log", "cauchy"]

@type_check_only
class QuadOpts(TypedDict, total=False):
    epsabs: _FloatLike
    epsrel: _FloatLike
    limit: _IntLike
    points: _ArrayLikeFloat_co
    weight: QuadWeights
    wvar: _FloatLike | tuple[_FloatLike, _FloatLike]
    wopts: tuple[_IntLike, npt.NDArray[np.float32 | np.float64]]

@type_check_only
class QuadInfoDict(TypedDict):
    neval: int
    last: int
    alist: onp.Array1D[np.float64]
    blist: onp.Array1D[np.float64]
    rlist: onp.Array1D[np.float64]
    elist: onp.Array1D[np.float64]
    iord: onp.Array1D[np.int_]

    # if `points` is provided
    pts: NotRequired[onp.Array1D[np.float64]]
    level: NotRequired[onp.Array1D[np.int_]]
    ndin: NotRequired[onp.Array1D[np.int_]]

    # finite integration limits
    momcom: NotRequired[float | np.float64]
    nnlog: NotRequired[onp.Array1D[np.int_]]
    chebmo: NotRequired[onp.Array[tuple[Literal[25], int], np.int_]]

    # single infitite integration limit and numerical error
    lst: NotRequired[int]
    rslst: NotRequired[onp.Array1D[np.float64]]
    erlst: NotRequired[onp.Array1D[np.float64]]
    ierlst: NotRequired[onp.Array1D[np.float64]]

@type_check_only
class ODEInfoDict(TypedDict):
    hu: onp.Array1D[np.float64]
    tcur: onp.Array1D[np.float64]
    tolsf: onp.Array1D[np.float64]
    tsw: float
    nst: int
    nfe: int
    nje: int
    nqu: onp.Array1D[np.int_]
    imxer: int
    lenrw: int
    leniw: int
    mused: onp.Array1D[np.int_]
