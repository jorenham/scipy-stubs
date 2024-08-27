from collections.abc import Callable, Sequence
from typing import Any, Final, Generic, Literal, TypeAlias, overload

import numpy as np
import numpy.typing as npt
import optype.numpy as onpt
from typing_extensions import LiteralString, Self, TypeVar

import scipy._typing as spt
from scipy._typing import Untyped

__all__ = [
    "_ShapeInfo",
    "argsreduce",
    "get_distribution_names",
    "rv_continuous",
    "rv_continuous_frozen",
    "rv_discrete",
    "rv_discrete_frozen",
    "rv_frozen",
    "rv_generic",
    "rv_sample",
]

_RVG_co = TypeVar("_RVG_co", bound=rv_generic, covariant=True, default=rv_generic)
_RVC_co = TypeVar("_RVC_co", bound=rv_continuous, covariant=True, default=rv_continuous)
_RVD_co = TypeVar("_RVD_co", bound=rv_discrete, covariant=True, default=rv_discrete)

# scalar-like that is coerceable (indicated by the `_in` suffix) to a `numpy.float64`
_Scalar_f8_in: TypeAlias = np.float64 | np.float32 | np.float16 | np.integer[Any] | np.bool_
_AnyScalar_f8_in: TypeAlias = float | _Scalar_f8_in
_AnyArray_f8_in: TypeAlias = float | onpt.CanArray[tuple[int, ...], np.dtype[_Scalar_f8_in]] | Sequence[_AnyArray_f8_in]
_AnyArray_f8_out: TypeAlias = np.float64 | npt.NDArray[np.float64]

# numpy.random
_RNG: TypeAlias = np.random.Generator | np.random.RandomState
_Seed: TypeAlias = _RNG | int

_ArgT = TypeVar("_ArgT", bound=_AnyArray_f8_in, default=_AnyArray_f8_in)
# there are at most 4 + 2 args
_RVArgs: TypeAlias = (
    tuple[()]
    | tuple[_ArgT]
    | tuple[_ArgT, _ArgT]
    | tuple[_ArgT, _ArgT, _ArgT]
    | tuple[_ArgT, _ArgT, _ArgT, _ArgT]
    | tuple[_ArgT, _ArgT, _ArgT, _ArgT, _ArgT]
    | tuple[_ArgT, _ArgT, _ArgT, _ArgT, _ArgT, _ArgT]
)
_RVKwds: TypeAlias = dict[str, _AnyArray_f8_in]

###

parse_arg_template: Final[str]

def argsreduce(cond: npt.NDArray[np.bool_], *args: npt.ArrayLike) -> list[npt.NDArray[np.floating[Any] | np.integer[Any]]]: ...

class rv_frozen(Generic[_RVG_co]):
    args: Final[_RVArgs]
    kwds: Final[_RVKwds]
    @property
    def dist(self) -> _RVG_co: ...
    @property
    def random_state(self) -> _RNG: ...
    @random_state.setter
    def random_state(self, seed: _Seed, /) -> None: ...
    def __init__(self, dist: _RVG_co, *args: _AnyArray_f8_in, **kwds: _AnyArray_f8_in) -> None: ...
    def cdf(self, x: _AnyArray_f8_in) -> _AnyArray_f8_out: ...
    def logcdf(self, x: _AnyArray_f8_in) -> _AnyArray_f8_out: ...
    def ppf(self, q: _AnyArray_f8_in) -> _AnyArray_f8_out: ...
    def isf(self, q: _AnyArray_f8_in) -> _AnyArray_f8_out: ...
    def rvs(
        self,
        size: spt.AnyInt | tuple[spt.AnyInt, ...] | None = None,
        random_state: _Seed | None = None,
    ) -> _AnyArray_f8_out: ...
    def sf(self, x: _AnyArray_f8_in) -> _AnyArray_f8_out: ...
    def logsf(self, x: _AnyArray_f8_in) -> _AnyArray_f8_out: ...
    @overload
    def stats(
        self,
        moments: Literal["m", "v", "s", "k"],
    ) -> tuple[np.float64] | tuple[npt.NDArray[np.float64]]: ...
    @overload
    def stats(
        self,
        moments: Literal["mv", "ms", "mk", "vs", "vk", "sk"] = ...,
    ) -> tuple[np.float64, np.float64] | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
    @overload
    def stats(
        self,
        moments: Literal["mvs", "mvk", "msk", "vsk"],
    ) -> (
        tuple[np.float64, np.float64, np.float64]
        | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]
    ): ...
    @overload
    def stats(
        self,
        moments: Literal["mvsk"],
    ) -> (
        tuple[np.float64, np.float64, np.float64, np.float64]
        | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]
    ): ...
    def median(self) -> _AnyArray_f8_out: ...
    def mean(self) -> _AnyArray_f8_out: ...
    def var(self) -> _AnyArray_f8_out: ...
    def std(self) -> _AnyArray_f8_out: ...
    # order defaults to `None`, but that will `raise TypeError`
    def moment(self, order: int) -> Untyped: ...
    def entropy(self) -> _AnyArray_f8_out: ...
    def interval(
        self,
        confidence: _AnyScalar_f8_in | None = None,
    ) -> tuple[np.float64, np.float64] | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
    def support(self) -> tuple[np.float64, np.float64] | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
    # requires all args to be scalars
    def expect(
        self,
        func: Callable[[float], float] | None = None,
        lb: _AnyScalar_f8_in | None = None,
        ub: _AnyScalar_f8_in | None = None,
        conditional: bool = False,
        # TODO: use `TypedDict` and `Unpack` with the `scipy.integrate.quad` kwargs
        **kwds: Any,
    ) -> np.float64: ...

class rv_continuous_frozen(rv_frozen[_RVC_co], Generic[_RVC_co]):
    def pdf(self, x: _AnyArray_f8_in) -> _AnyArray_f8_out: ...
    def logpdf(self, x: _AnyArray_f8_in) -> _AnyArray_f8_out: ...

class rv_discrete_frozen(rv_frozen[_RVD_co], Generic[_RVD_co]):
    def pmf(self, k: _AnyArray_f8_in) -> _AnyArray_f8_out: ...
    def logpmf(self, k: _AnyArray_f8_in) -> _AnyArray_f8_out: ...

class rv_generic:
    @property
    def random_state(self) -> _RNG: ...
    @random_state.setter
    def random_state(self, seed: _Seed, /) -> None: ...
    def __init__(self, seed: _Seed | None = None) -> None: ...
    def __call__(self, *args: _AnyArray_f8_in, **kwds: _AnyArray_f8_in) -> rv_frozen[Self]: ...
    def freeze(self, *args: _AnyArray_f8_in, **kwds: _AnyArray_f8_in) -> rv_frozen[Self]: ...
    @overload
    def rvs(
        self,
        *args: npt.ArrayLike,
        random_state: _Seed,
        discrete: Literal[True],
        **kwds: _AnyArray_f8_in,
    ) -> int | npt.NDArray[np.int64]: ...
    @overload
    def rvs(
        self,
        *args: npt.ArrayLike,
        random_state: _Seed,
        discrete: Literal[False, None] = ...,
        **kwds: _AnyArray_f8_in,
    ) -> np.float64 | npt.NDArray[np.float64]: ...
    @overload
    def stats(
        self,
        *args: _AnyScalar_f8_in,
        moment: Literal["m", "v", "s", "k", "mv", "ms", "mk", "vs", "vk", "sk", "mvs", "mvk", "msk", "vsk", "mvsk"] = ...,
        **kwds: _AnyScalar_f8_in,
    ) -> tuple[np.float64, ...]: ...
    @overload
    def stats(
        self,
        *args: _AnyArray_f8_in,
        moment: Literal["m", "v", "s", "k", "mv", "ms", "mk", "vs", "vk", "sk", "mvs", "mvk", "msk", "vsk", "mvsk"] = ...,
        **kwds: _AnyArray_f8_in,
    ) -> tuple[np.float64, ...] | tuple[npt.NDArray[np.float64], ...]: ...
    @overload
    def entropy(self) -> np.float64: ...
    @overload
    def entropy(self, *args: _AnyScalar_f8_in, **kwds: _AnyScalar_f8_in) -> np.float64: ...
    @overload
    def entropy(self, *args: _AnyArray_f8_in, **kwds: _AnyArray_f8_in) -> _AnyArray_f8_out: ...
    @overload
    def moment(self, order: int, /, *args: _AnyScalar_f8_in, **kwds: _AnyScalar_f8_in) -> np.float64: ...
    @overload
    def moment(self, order: int, /, *args: _AnyArray_f8_in, **kwds: _AnyArray_f8_in) -> _AnyArray_f8_out: ...
    @overload
    def median(self, *args: _AnyScalar_f8_in, **kwds: _AnyScalar_f8_in) -> np.float64: ...
    @overload
    def median(self, *args: _AnyArray_f8_in, **kwds: _AnyArray_f8_in) -> _AnyArray_f8_out: ...
    @overload
    def mean(self, *args: _AnyScalar_f8_in, **kwds: _AnyScalar_f8_in) -> np.float64: ...
    @overload
    def mean(self, *args: _AnyArray_f8_in, **kwds: _AnyArray_f8_in) -> _AnyArray_f8_out: ...
    @overload
    def var(self, *args: _AnyScalar_f8_in, **kwds: _AnyScalar_f8_in) -> np.float64: ...
    @overload
    def var(self, *args: _AnyArray_f8_in, **kwds: _AnyArray_f8_in) -> _AnyArray_f8_out: ...
    @overload
    def std(self, *args: _AnyScalar_f8_in, **kwds: _AnyScalar_f8_in) -> np.float64: ...
    @overload
    def std(self, *args: _AnyArray_f8_in, **kwds: _AnyArray_f8_in) -> _AnyArray_f8_out: ...
    @overload
    def interval(
        self,
        confidence: _AnyScalar_f8_in,
        /,
        *args: _AnyScalar_f8_in,
        **kwds: _AnyScalar_f8_in,
    ) -> tuple[np.float64, np.float64]: ...
    @overload
    def interval(
        self,
        confidence: _AnyArray_f8_out,
        /,
        *args: _AnyArray_f8_out,
        **kwds: _AnyArray_f8_out,
    ) -> tuple[np.float64, np.float64] | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
    @overload
    def support(self, *args: _AnyScalar_f8_in, **kwds: _AnyScalar_f8_in) -> tuple[np.float64, np.float64]: ...
    @overload
    def support(
        self,
        *args: _AnyArray_f8_in,
        **kwds: _AnyArray_f8_in,
    ) -> tuple[np.float64, np.float64] | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
    def nnlf(self, theta: Sequence[_Scalar_f8_in], x: onpt.AnyNumberArray) -> np.float64: ...

class _ShapeInfo:
    name: Final[LiteralString]
    integrality: Final[bool]
    domain: Final[Sequence[float]]  # in practice always a list of size two
    def __init__(
        self,
        name: LiteralString,
        integrality: bool = False,
        domain: Sequence[float] = ...,
        inclusive: Sequence[bool] = (True, True),
    ) -> None: ...

class rv_continuous(rv_generic):
    moment_type: Final[Literal[0, 1]]
    name: Final[LiteralString]
    # defaults to `-inf`
    a: Final[float]
    # defaults to `+inf`
    b: Final[float]
    # defaults to `nan`
    badvalue: Final[float]
    # defaults to 1e-14
    xtol: Final[float]
    # comma-separated names of shape parameters
    shapes: Final[LiteralString]

    def __init__(
        self,
        *,
        name: LiteralString,
        momtype: Literal[0, 1] = 1,
        a: float | None = None,
        b: float | None = None,
        xtol: float = 1e-14,
        badvalue: float | None = None,
        longname: LiteralString | None = None,
        shapes: LiteralString | None = None,
        seed: _Seed | None = None,
    ) -> None: ...
    @overload
    def pdf(self, x: _Scalar_f8_in, *args: _Scalar_f8_in, **kwds: _Scalar_f8_in) -> np.float64: ...
    @overload
    def pdf(
        self,
        x: _AnyArray_f8_in,
        *args: _AnyArray_f8_in,
        **kwds: _AnyArray_f8_in,
    ) -> np.float64 | npt.NDArray[np.float64]: ...
    @overload
    def logpdf(self, x: _Scalar_f8_in, *args: _Scalar_f8_in, **kwds: _Scalar_f8_in) -> np.float64: ...
    @overload
    def logpdf(
        self,
        x: _AnyArray_f8_in,
        *args: _AnyArray_f8_in,
        **kwds: _AnyArray_f8_in,
    ) -> np.float64 | npt.NDArray[np.float64]: ...
    @overload
    def cdf(self, x: _Scalar_f8_in, *args: _Scalar_f8_in, **kwds: _Scalar_f8_in) -> np.float64: ...
    @overload
    def cdf(
        self,
        x: _AnyArray_f8_in,
        *args: _AnyArray_f8_in,
        **kwds: _AnyArray_f8_in,
    ) -> np.float64 | npt.NDArray[np.float64]: ...
    @overload
    def logcdf(self, x: _Scalar_f8_in, *args: _Scalar_f8_in, **kwds: _Scalar_f8_in) -> np.float64: ...
    @overload
    def logcdf(
        self,
        x: _AnyArray_f8_in,
        *args: _AnyArray_f8_in,
        **kwds: _AnyArray_f8_in,
    ) -> np.float64 | npt.NDArray[np.float64]: ...
    @overload
    def sf(self, x: _Scalar_f8_in, *args: _Scalar_f8_in, **kwds: _Scalar_f8_in) -> np.float64: ...
    @overload
    def sf(
        self,
        x: _AnyArray_f8_in,
        *args: _AnyArray_f8_in,
        **kwds: _AnyArray_f8_in,
    ) -> np.float64 | npt.NDArray[np.float64]: ...
    @overload
    def logsf(self, x: _Scalar_f8_in, *args: _Scalar_f8_in, **kwds: _Scalar_f8_in) -> np.float64: ...
    @overload
    def logsf(
        self,
        x: _AnyArray_f8_in,
        *args: _AnyArray_f8_in,
        **kwds: _AnyArray_f8_in,
    ) -> np.float64 | npt.NDArray[np.float64]: ...
    @overload
    def ppf(self, q: _Scalar_f8_in, *args: _Scalar_f8_in, **kwds: _Scalar_f8_in) -> np.float64: ...
    @overload
    def ppf(self, q: _AnyArray_f8_in, *args: _AnyArray_f8_in, **kwds: _AnyArray_f8_in) -> _AnyArray_f8_out: ...
    @overload
    def isf(self, q: _Scalar_f8_in, *args: _Scalar_f8_in, **kwds: _Scalar_f8_in) -> np.float64: ...
    @overload
    def isf(self, q: _AnyArray_f8_in, *args: _AnyArray_f8_in, **kwds: _AnyArray_f8_in) -> _AnyArray_f8_out: ...
    def fit(
        self,
        data: _AnyArray_f8_in,
        *args: _Scalar_f8_in,
        optimizer: Callable[
            [npt.NDArray[np.float64], tuple[np.float64, ...], tuple[np.float64, ...], bool], tuple[np.float64, ...]
        ],
        method: Literal["MLE", "MM"] = "MLE",
        **kwds: _Scalar_f8_in,
    ) -> tuple[np.float64, ...]: ...
    def fit_loc_scale(self, data: _AnyArray_f8_in, *args: _Scalar_f8_in) -> Untyped: ...
    def expect(
        self,
        func: Callable[[float], float] | None = None,
        args: tuple[float, ...] = (),
        loc: int = 0,
        scale: int = 1,
        lb: _AnyScalar_f8_in | None = None,
        ub: _AnyScalar_f8_in | None = None,
        conditional: bool = False,
        # TODO: use `TypedDict` and `Unpack` with the `scipy.integrate.quad` kwargs
        **kwds: Any,
    ) -> Untyped: ...

class rv_discrete(rv_generic):
    def __new__(
        cls,
        a: int = 0,
        b=...,
        name: Untyped | None = None,
        badvalue: Untyped | None = None,
        moment_tol: float = 1e-08,
        values: Untyped | None = None,
        inc: int = 1,
        longname: Untyped | None = None,
        shapes: Untyped | None = None,
        seed: Untyped | None = None,
    ) -> Untyped: ...
    badvalue: Untyped
    a: Untyped
    b: Untyped
    moment_tol: Untyped
    inc: Untyped
    shapes: Untyped
    def __init__(
        self,
        a: int = 0,
        b=...,
        name: Untyped | None = None,
        badvalue: Untyped | None = None,
        moment_tol: float = 1e-08,
        values: Untyped | None = None,
        inc: int = 1,
        longname: Untyped | None = None,
        shapes: Untyped | None = None,
        seed: Untyped | None = None,
    ): ...
    def rvs(self, *args, **kwargs) -> Untyped: ...
    def pmf(self, k, *args, **kwds) -> Untyped: ...
    def logpmf(self, k, *args, **kwds) -> Untyped: ...
    def cdf(self, k, *args, **kwds) -> Untyped: ...
    def logcdf(self, k, *args, **kwds) -> Untyped: ...
    def sf(self, k, *args, **kwds) -> Untyped: ...
    def logsf(self, k, *args, **kwds) -> Untyped: ...
    def ppf(self, q, *args, **kwds) -> Untyped: ...
    def isf(self, q, *args, **kwds) -> Untyped: ...
    def expect(
        self,
        func: Untyped | None = None,
        args=(),
        loc: int = 0,
        lb: Untyped | None = None,
        ub: Untyped | None = None,
        conditional: bool = False,
        maxcount: int = 1000,
        tolerance: float = 1e-10,
        chunksize: int = 32,
    ) -> Untyped: ...

class rv_sample(rv_discrete):
    badvalue: Untyped
    moment_tol: Untyped
    inc: Untyped
    shapes: Untyped
    vecentropy: Untyped
    xk: Untyped
    pk: Untyped
    a: Untyped
    b: Untyped
    qvals: Untyped
    def __init__(
        self,
        a: int = 0,
        b=...,
        name: Untyped | None = None,
        badvalue: Untyped | None = None,
        moment_tol: float = 1e-08,
        values: Untyped | None = None,
        inc: int = 1,
        longname: Untyped | None = None,
        shapes: Untyped | None = None,
        seed: Untyped | None = None,
    ): ...
    def generic_moment(self, n) -> Untyped: ...

def get_distribution_names(namespace_pairs, rv_base_class) -> Untyped: ...
