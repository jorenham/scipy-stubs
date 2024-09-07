import abc
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Final, Generic, Literal, TypeAlias, overload, type_check_only
from typing_extensions import LiteralString, Self, TypeVar, Unpack, override

import numpy as np
import numpy.typing as npt
import optype.numpy as onpt
from numpy._typing import _ArrayLikeInt_co
import scipy._typing as spt
from scipy.integrate._typing import QuadOpts as _QuadOpts

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

_XT_co = TypeVar("_XT_co", bound=np.number[Any], covariant=True, default=np.number[Any])
_PT_co = TypeVar("_PT_co", bound=np.floating[Any], covariant=True, default=np.float32 | np.float64)

_ShapeT = TypeVar("_ShapeT", bound=tuple[int, ...], default=tuple[int, ...])

_T = TypeVar("_T")
_Tuple2: TypeAlias = tuple[_T, _T]
_Tuple3: TypeAlias = tuple[_T, _T, _T]
_Tuple4: TypeAlias = tuple[_T, _T, _T, _T]

_Scalar_f8_co: TypeAlias = np.float64 | np.float32 | np.float16 | np.integer[Any] | np.bool_
_ScalarLike_f8_co: TypeAlias = float | _Scalar_f8_co

_Array_f8: TypeAlias = onpt.Array[_ShapeT, np.float64]
_Array_f8_co: TypeAlias = onpt.Array[_ShapeT, _Scalar_f8_co]
_ArrayLike_f8_co: TypeAlias = float | onpt.CanArray[tuple[int, ...], np.dtype[_Scalar_f8_co]] | Sequence[_ArrayLike_f8_co]

_SCT = TypeVar("_SCT", bound=np.generic)
_ScalarOrArray: TypeAlias = _SCT | onpt.Array[tuple[int, ...], _SCT]
_ScalarOrArray_f8: TypeAlias = _ScalarOrArray[np.float64]

_ArgT = TypeVar("_ArgT", bound=_ArrayLike_f8_co, default=_ArrayLike_f8_co)
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
_RVKwds: TypeAlias = dict[str, _ArrayLike_f8_co]

_StatsMoment: TypeAlias = Literal["m", "v", "s", "k", "mv", "ms", "mk", "vs", "vk", "sk", "mvs", "mvk", "msk", "vsk", "mvsk"]
_FitMethod: TypeAlias = Literal["MLE", "MM"]

###

docheaders: Final[dict[str, str]] = ...
docdict: Final[dict[str, str]] = ...
docdict_discrete: Final[dict[str, str]] = ...
parse_arg_template: Final[str] = ...

def argsreduce(cond: npt.NDArray[np.bool_], *args: _ArrayLike_f8_co) -> list[_Array_f8_co]: ...

class rv_frozen(Generic[_RVG_co]):
    args: Final[_RVArgs]
    kwds: Final[_RVKwds]
    dist: _RVG_co
    @property
    def random_state(self, /) -> spt.RNG: ...
    @random_state.setter
    def random_state(self, seed: spt.Seed, /) -> None: ...
    def __init__(self, /, dist: _RVG_co, *args: _ArrayLike_f8_co, **kwds: _ArrayLike_f8_co) -> None: ...
    def cdf(self, /, x: _ArrayLike_f8_co) -> _ScalarOrArray_f8: ...
    def logcdf(self, /, x: _ArrayLike_f8_co) -> _ScalarOrArray_f8: ...
    def ppf(self, /, q: _ArrayLike_f8_co) -> _ScalarOrArray_f8: ...
    def isf(self, /, q: _ArrayLike_f8_co) -> _ScalarOrArray_f8: ...
    def rvs(
        self,
        /,
        size: spt.AnyInt | tuple[spt.AnyInt, ...] | None = None,
        random_state: spt.Seed | None = None,
    ) -> _ScalarOrArray_f8: ...
    def sf(self, /, x: _ArrayLike_f8_co) -> _ScalarOrArray_f8: ...
    def logsf(self, /, x: _ArrayLike_f8_co) -> _ScalarOrArray_f8: ...
    @overload
    def stats(self, /, moments: Literal["m", "v", "s", "k"]) -> tuple[np.float64] | tuple[_Array_f8]: ...
    @overload
    def stats(
        self,
        /,
        moments: Literal["mv", "ms", "mk", "vs", "vk", "sk"] = ...,
    ) -> _Tuple2[np.float64] | _Tuple2[_Array_f8]: ...
    @overload
    def stats(self, /, moments: Literal["mvs", "mvk", "msk", "vsk"]) -> _Tuple3[np.float64] | _Tuple3[_Array_f8]: ...
    @overload
    def stats(self, /, moments: Literal["mvsk"]) -> _Tuple4[np.float64] | _Tuple4[_Array_f8]: ...
    def median(self, /) -> _ScalarOrArray_f8: ...
    def mean(self, /) -> _ScalarOrArray_f8: ...
    def var(self, /) -> _ScalarOrArray_f8: ...
    def std(self, /) -> _ScalarOrArray_f8: ...
    # order defaults to `None`, but that will `raise TypeError`
    def moment(self, /, order: int | None = None) -> np.float64: ...
    def entropy(self, /) -> _ScalarOrArray_f8: ...
    def interval(self, /, confidence: _ScalarLike_f8_co | None = None) -> _Tuple2[np.float64] | _Tuple2[_Array_f8]: ...
    def support(self, /) -> _Tuple2[np.float64] | _Tuple2[_Array_f8]: ...
    # requires all args to be scalars
    def expect(
        self,
        /,
        func: Callable[[float], float] | None = None,
        lb: _ScalarLike_f8_co | None = None,
        ub: _ScalarLike_f8_co | None = None,
        conditional: bool = False,
        **kwds: Unpack[_QuadOpts],
    ) -> np.float64: ...

class rv_continuous_frozen(rv_frozen[_RVC_co], Generic[_RVC_co]):
    def pdf(self, /, x: _ArrayLike_f8_co) -> _ScalarOrArray_f8: ...
    def logpdf(self, /, x: _ArrayLike_f8_co) -> _ScalarOrArray_f8: ...

class rv_discrete_frozen(rv_frozen[_RVD_co], Generic[_RVD_co]):
    def pmf(self, /, k: _ArrayLike_f8_co) -> _ScalarOrArray_f8: ...
    def logpmf(self, /, k: _ArrayLike_f8_co) -> _ScalarOrArray_f8: ...

# NOTE: Because of the limitations of `ParamSpec`, there is no proper way to annotate specific "positional or keyword arguments".
# Considering the Liskov Substitution Principle, the only remaining option is to annotate `*args, and `**kwargs` as `Any`.
class rv_generic:
    # TODO: private methods
    def __init__(self, /, seed: spt.Seed | None = None) -> None: ...
    @property
    def random_state(self, /) -> spt.RNG: ...
    @random_state.setter
    def random_state(self, seed: spt.Seed, /) -> None: ...
    @abc.abstractmethod
    def _attach_methods(self, /) -> None: ...
    def _attach_argparser_methods(self, /) -> None: ...
    def _construct_argparser(
        self, /, meths_to_inspect: Iterable[Callable[..., Any]], locscale_in: str, locscale_out: str
    ) -> None: ...
    def _construct_doc(self, /, docdict: dict[str, str], shapes_vals: tuple[float, ...] | None = None) -> None: ...
    def _construct_default_doc(
        self,
        /,
        longname: str | None = None,
        docdict: dict[str, str] | None = None,
        discrete: Literal["continuous", "discrete"] = "continuous",
    ) -> None: ...
    def freeze(self, /, *args: _ArrayLike_f8_co, **kwds: _ArrayLike_f8_co) -> rv_frozen[Self]: ...
    def __call__(self, /, *args: Any, **kwds: Any) -> rv_frozen[Self]: ...
    def _stats(self, /, *args: Any, **kwds: Any) -> _Tuple4[_ScalarOrArray_f8 | None]: ...
    def _munp(self, /, n: _ArrayLikeInt_co, *args: Any) -> _Array_f8: ...
    def _argcheck_rvs(
        self,
        /,
        *args: Any,
        size: _ArrayLikeInt_co | None = None,
    ) -> tuple[list[_Array_f8_co], _Array_f8_co, _Array_f8_co, tuple[int, ...] | tuple[np.int_, ...]]: ...
    def _argcheck(self, /, *args: Any) -> _ScalarOrArray[np.bool_]: ...
    def _get_support(self, /, *args: Any, **kwargs: Any) -> _Tuple2[np.float64] | _Tuple2[_Array_f8]: ...
    def _support_mask(self, /, x: _Array_f8_co, *args: Any) -> _ScalarOrArray[np.bool_]: ...
    def _open_support_mask(self, /, x: _Array_f8_co, *args: Any) -> _ScalarOrArray[np.bool_]: ...
    def _rvs(
        self,
        /,
        *args: Any,
        size: spt.AnyInt | tuple[spt.AnyInt, ...] | None = None,
        random_state: spt.Seed | None = None,
    ) -> _ScalarOrArray_f8: ...
    def _logcdf(self, /, x: _ScalarOrArray[_Scalar_f8_co], *args: Any) -> _ScalarOrArray_f8: ...
    def _sf(self, /, x: _ScalarOrArray[_Scalar_f8_co], *args: Any) -> _ScalarOrArray_f8: ...
    def _logsf(self, /, x: _ScalarOrArray[_Scalar_f8_co], *args: Any) -> _ScalarOrArray_f8: ...
    def _ppf(self, /, q: _ScalarOrArray[_Scalar_f8_co], *args: Any) -> _ScalarOrArray_f8: ...
    def _isf(self, /, q: _ScalarOrArray[_Scalar_f8_co], *args: Any) -> _ScalarOrArray_f8: ...
    @overload
    def rvs(
        self,
        /,
        *args: _ScalarLike_f8_co,
        random_state: spt.Seed,
        discrete: Literal[True, 1],
        **kwds: _ArrayLike_f8_co,
    ) -> int | _ScalarOrArray[np.int64]: ...  # NOTE: this is `int64`; not `int_`
    @overload
    def rvs(
        self,
        /,
        *args: _ScalarLike_f8_co,
        random_state: spt.Seed,
        discrete: Literal[False, 0, None] = ...,
        **kwds: _ArrayLike_f8_co,
    ) -> _ScalarOrArray_f8: ...
    @overload
    def stats(
        self,
        /,
        *args: _ScalarLike_f8_co,
        moment: _StatsMoment = ...,
        **kwds: _ScalarLike_f8_co,
    ) -> tuple[np.float64, ...]: ...
    @overload
    def stats(
        self,
        /,
        *args: _ArrayLike_f8_co,
        moment: _StatsMoment = ...,
        **kwds: _ArrayLike_f8_co,
    ) -> tuple[np.float64, ...] | tuple[_Array_f8, ...]: ...
    @overload
    def entropy(self, /) -> np.float64: ...
    @overload
    def entropy(self, /, *args: _ScalarLike_f8_co, **kwds: _ScalarLike_f8_co) -> np.float64: ...
    @overload
    def entropy(self, /, *args: _ArrayLike_f8_co, **kwds: _ArrayLike_f8_co) -> _ScalarOrArray_f8: ...
    @overload
    def moment(self, /, order: spt.AnyInt, *args: _ScalarLike_f8_co, **kwds: _ScalarLike_f8_co) -> np.float64: ...
    @overload
    def moment(self, /, order: spt.AnyInt, *args: _ArrayLike_f8_co, **kwds: _ArrayLike_f8_co) -> _ScalarOrArray_f8: ...
    @overload
    def median(self, /, *args: _ScalarLike_f8_co, **kwds: _ScalarLike_f8_co) -> np.float64: ...
    @overload
    def median(self, /, *args: _ArrayLike_f8_co, **kwds: _ArrayLike_f8_co) -> _ScalarOrArray_f8: ...
    @overload
    def mean(self, /, *args: _ScalarLike_f8_co, **kwds: _ScalarLike_f8_co) -> np.float64: ...
    @overload
    def mean(self, /, *args: _ArrayLike_f8_co, **kwds: _ArrayLike_f8_co) -> _ScalarOrArray_f8: ...
    @overload
    def var(self, /, *args: _ScalarLike_f8_co, **kwds: _ScalarLike_f8_co) -> np.float64: ...
    @overload
    def var(self, /, *args: _ArrayLike_f8_co, **kwds: _ArrayLike_f8_co) -> _ScalarOrArray_f8: ...
    @overload
    def std(self, /, *args: _ScalarLike_f8_co, **kwds: _ScalarLike_f8_co) -> np.float64: ...
    @overload
    def std(self, /, *args: _ArrayLike_f8_co, **kwds: _ArrayLike_f8_co) -> _ScalarOrArray_f8: ...
    @overload
    def interval(
        self,
        /,
        confidence: _ScalarLike_f8_co,
        *args: _ScalarLike_f8_co,
        **kwds: _ScalarLike_f8_co,
    ) -> _Tuple2[np.float64]: ...
    @overload
    def interval(
        self,
        /,
        confidence: _ScalarOrArray_f8,
        *args: _ScalarOrArray_f8,
        **kwds: _ScalarOrArray_f8,
    ) -> _Tuple2[np.float64] | _Tuple2[_Array_f8]: ...
    @overload
    def support(self, /, *args: _ScalarLike_f8_co, **kwds: _ScalarLike_f8_co) -> _Tuple2[np.float64]: ...
    @overload
    def support(
        self,
        /,
        *args: _ArrayLike_f8_co,
        **kwds: _ArrayLike_f8_co,
    ) -> _Tuple2[np.float64] | _Tuple2[_Array_f8]: ...
    def nnlf(self, /, theta: Sequence[_ScalarLike_f8_co], x: _ArrayLike_f8_co) -> _ScalarOrArray_f8: ...
    def _nnlf(self, /, x: npt.NDArray[np.floating[Any]], *args: Any) -> _ScalarOrArray_f8: ...
    def _penalized_nnlf(self, /, theta: Sequence[Any], x: _Array_f8_co) -> np.float64: ...
    def _penalized_nlpsf(self, /, theta: Sequence[Any], x: _Array_f8_co) -> np.float64: ...

class _ShapeInfo:
    name: Final[LiteralString]
    integrality: Final[bool]
    domain: Final[Sequence[float]]  # in practice always a list of size two
    def __init__(
        self,
        /,
        name: LiteralString,
        integrality: bool = False,
        domain: Sequence[float] = ...,
        inclusive: Sequence[bool] = (True, True),
    ) -> None: ...

@type_check_only
class _rv_mixin:
    name: Final[LiteralString]
    a: Final[float]
    b: Final[float]
    badvalue: Final[float]
    shapes: Final[LiteralString]

    def _attach_methods(self, /) -> None: ...
    def generic_moment(self, /, n: _ArrayLikeInt_co, *args: _ScalarLike_f8_co) -> _Array_f8: ...
    def _logpxf(self, /, x: _Array_f8_co, *args: Any) -> _Array_f8: ...
    def _cdf_single(self, /, x: _ScalarLike_f8_co, *args: Any) -> np.float64: ...
    def _cdfvec(self, /, x: _Array_f8_co, *args: Any) -> _Array_f8: ...
    def _cdf(self, /, x: _Array_f8_co, *args: Any) -> _Array_f8: ...
    def _ppfvec(self, /, q: _Array_f8_co, *args: Any) -> _Array_f8: ...
    def _unpack_loc_scale(
        self,
        /,
        theta: Sequence[_ArrayLike_f8_co],
    ) -> tuple[_ArrayLike_f8_co, _ArrayLike_f8_co, tuple[_ArrayLike_f8_co]]: ...

class rv_continuous(_rv_mixin, rv_generic):
    moment_type: Final[Literal[0, 1]]
    xtol: Final[float]

    def __init__(
        self,
        /,
        momtype: Literal[0, 1] = 1,
        a: float | None = None,
        b: float | None = None,
        xtol: float = 1e-14,
        badvalue: float | None = None,
        name: LiteralString | None = None,
        longname: LiteralString | None = None,
        shapes: LiteralString | None = None,
        seed: spt.Seed | None = None,
    ) -> None: ...
    @override
    def __call__(
        self,
        /,
        *args: _ArrayLike_f8_co,
        loc: _ArrayLike_f8_co = 0,
        scale: _ArrayLike_f8_co = 1,
        **kwds: _ArrayLike_f8_co,
    ) -> rv_continuous_frozen[Self]: ...
    @override
    def freeze(
        self,
        /,
        *args: _ArrayLike_f8_co,
        loc: _ArrayLike_f8_co = 0,
        scale: _ArrayLike_f8_co = 1,
        **kwds: _ArrayLike_f8_co,
    ) -> rv_continuous_frozen[Self]: ...
    def _pdf(self, /, x: _Array_f8_co, *args: Any) -> _Array_f8: ...
    def _logpdf(self, /, x: _Array_f8_co, *args: Any) -> _Array_f8: ...
    @overload
    def pdf(
        self,
        /,
        x: _ScalarLike_f8_co,
        *args: _ScalarLike_f8_co,
        loc: _ScalarLike_f8_co = 0,
        scale: _ScalarLike_f8_co = 1,
        **kwds: _ScalarLike_f8_co,
    ) -> np.float64: ...
    @overload
    def pdf(
        self,
        /,
        x: _ArrayLike_f8_co,
        *args: _ArrayLike_f8_co,
        loc: _ArrayLike_f8_co = 0,
        scale: _ArrayLike_f8_co = 1,
        **kwds: _ArrayLike_f8_co,
    ) -> _ScalarOrArray_f8: ...
    @overload
    def logpdf(
        self,
        /,
        x: _ScalarLike_f8_co,
        *args: _ScalarLike_f8_co,
        loc: _ScalarLike_f8_co = 0,
        scale: _ScalarLike_f8_co = 1,
        **kwds: _ScalarLike_f8_co,
    ) -> np.float64: ...
    @overload
    def logpdf(
        self,
        /,
        x: _ArrayLike_f8_co,
        *args: _ArrayLike_f8_co,
        loc: _ArrayLike_f8_co = 0,
        scale: _ArrayLike_f8_co = 1,
        **kwds: _ArrayLike_f8_co,
    ) -> _ScalarOrArray_f8: ...
    @overload
    def cdf(
        self,
        /,
        x: _ScalarLike_f8_co,
        *args: _ScalarLike_f8_co,
        loc: _ScalarLike_f8_co = 0,
        scale: _ScalarLike_f8_co = 1,
        **kwds: _ScalarLike_f8_co,
    ) -> np.float64: ...
    @overload
    def cdf(
        self,
        /,
        x: _ArrayLike_f8_co,
        *args: _ArrayLike_f8_co,
        loc: _ArrayLike_f8_co = 0,
        scale: _ArrayLike_f8_co = 1,
        **kwds: _ArrayLike_f8_co,
    ) -> _ScalarOrArray_f8: ...
    @overload
    def logcdf(
        self,
        /,
        x: _ScalarLike_f8_co,
        *args: _ScalarLike_f8_co,
        loc: _ScalarLike_f8_co = 0,
        scale: _ScalarLike_f8_co = 1,
        **kwds: _ScalarLike_f8_co,
    ) -> np.float64: ...
    @overload
    def logcdf(
        self,
        /,
        x: _ArrayLike_f8_co,
        *args: _ArrayLike_f8_co,
        loc: _ArrayLike_f8_co = 0,
        scale: _ArrayLike_f8_co = 1,
        **kwds: _ArrayLike_f8_co,
    ) -> _ScalarOrArray_f8: ...
    @overload
    def sf(
        self,
        /,
        x: _ScalarLike_f8_co,
        *args: _ScalarLike_f8_co,
        loc: _ScalarLike_f8_co = 0,
        scale: _ScalarLike_f8_co = 1,
        **kwds: _ScalarLike_f8_co,
    ) -> np.float64: ...
    @overload
    def sf(
        self,
        /,
        x: _ArrayLike_f8_co,
        *args: _ArrayLike_f8_co,
        loc: _ArrayLike_f8_co = 0,
        scale: _ArrayLike_f8_co = 1,
        **kwds: _ArrayLike_f8_co,
    ) -> _ScalarOrArray_f8: ...
    @overload
    def logsf(
        self,
        /,
        x: _ScalarLike_f8_co,
        *args: _ScalarLike_f8_co,
        loc: _ScalarLike_f8_co = 0,
        scale: _ScalarLike_f8_co = 1,
        **kwds: _ScalarLike_f8_co,
    ) -> np.float64: ...
    @overload
    def logsf(
        self,
        /,
        x: _ArrayLike_f8_co,
        *args: _ArrayLike_f8_co,
        loc: _ArrayLike_f8_co = 0,
        scale: _ArrayLike_f8_co = 1,
        **kwds: _ArrayLike_f8_co,
    ) -> _ScalarOrArray_f8: ...
    @overload
    def ppf(
        self,
        /,
        q: _ScalarLike_f8_co,
        *args: _ScalarLike_f8_co,
        loc: _ScalarLike_f8_co = 0,
        scale: _ScalarLike_f8_co = 1,
        **kwds: _ScalarLike_f8_co,
    ) -> np.float64: ...
    @overload
    def ppf(
        self,
        /,
        q: _ArrayLike_f8_co,
        *args: _ArrayLike_f8_co,
        loc: _ArrayLike_f8_co = 0,
        scale: _ArrayLike_f8_co = 1,
        **kwds: _ArrayLike_f8_co,
    ) -> _ScalarOrArray_f8: ...
    @overload
    def isf(
        self,
        /,
        q: _ScalarLike_f8_co,
        *args: _ScalarLike_f8_co,
        loc: _ScalarLike_f8_co = 0,
        scale: _ScalarLike_f8_co = 1,
        **kwds: _ScalarLike_f8_co,
    ) -> np.float64: ...
    @overload
    def isf(
        self,
        /,
        q: _ArrayLike_f8_co,
        *args: _ArrayLike_f8_co,
        loc: _ArrayLike_f8_co = 0,
        scale: _ArrayLike_f8_co = 1,
        **kwds: _ArrayLike_f8_co,
    ) -> _ScalarOrArray_f8: ...
    def _nnlf_and_penalty(self, /, x: _Array_f8_co, args: Sequence[Any]) -> np.float64: ...
    def _fitstart(
        self,
        /,
        data: _ArrayLike_f8_co,
        args: tuple[Any, ...] | None = None,
    ) -> tuple[Unpack[tuple[float, ...]], float, float]: ...
    def _reduce_func(
        self,
        /,
        args: tuple[Any, ...],
        kwds: dict[str, Any],
        data: _ArrayLike_f8_co | None = None,
    ) -> tuple[
        list[float | np.float64],
        Callable[[list[_ScalarLike_f8_co], _Array_f8_co], float | np.float64],
        Callable[[list[_ScalarLike_f8_co], _Array_f8_co], list[float | np.float64]],
        list[float | np.float64],
    ]: ...
    def _moment_error(self, /, theta: Sequence[Any], x: _Array_f8_co, data_moments: _Array_f8_co[tuple[int]]) -> np.float64: ...
    def fit(
        self,
        /,
        data: _ArrayLike_f8_co,
        *args: _ScalarLike_f8_co,
        optimizer: Callable[
            [_Array_f8, tuple[float | np.float64, ...], tuple[float | np.float64, ...], bool],
            tuple[float | np.float64, ...],
        ],
        method: _FitMethod = "MLE",
        **kwds: _ScalarLike_f8_co,
    ) -> tuple[float | np.float64, ...]: ...
    def _fit_loc_scale_support(self, /, data: _ArrayLike_f8_co, *args: Any) -> _Tuple2[np.intp] | _Tuple2[float | np.float64]: ...
    def fit_loc_scale(self, /, data: _ArrayLike_f8_co, *args: _ScalarLike_f8_co) -> _Tuple2[np.float64]: ...
    def expect(
        self,
        /,
        func: Callable[[float], float] | None = None,
        args: tuple[_ScalarLike_f8_co, ...] = (),
        loc: _ScalarLike_f8_co = 0,
        scale: _ScalarLike_f8_co = 1,
        lb: _ScalarLike_f8_co | None = None,
        ub: _ScalarLike_f8_co | None = None,
        conditional: bool = False,
        **kwds: Unpack[_QuadOpts],
    ) -> np.float64: ...
    @override
    def rvs(  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        /,
        *args: _ScalarLike_f8_co,
        loc: _ScalarLike_f8_co = 0,
        scale: _ScalarLike_f8_co = 1,
        size: spt.AnyInt | tuple[spt.AnyInt, ...] = 1,
        random_state: spt.Seed | None = None,
        **kwds: _ArrayLike_f8_co,
    ) -> _ScalarOrArray_f8: ...

class rv_discrete(_rv_mixin, rv_generic):
    inc: Final[int]
    moment_tol: Final[float]

    def __new__(
        cls,
        a: float = 0,
        b: float = ...,
        name: LiteralString | None = None,
        badvalue: float | None = None,
        moment_tol: float = 1e-08,
        values: _Tuple2[_ArrayLike_f8_co] | None = None,
        inc: int = 1,
        longname: LiteralString | None = None,
        shapes: LiteralString | None = None,
        seed: spt.Seed | None = None,
    ) -> Self: ...
    def __init__(  # pyright: ignore[reportInconsistentConstructor]
        self,
        /,
        a: float = 0,
        b: float = ...,
        name: LiteralString | None = None,
        badvalue: float | None = None,
        moment_tol: float = 1e-08,
        values: None = None,
        inc: int = 1,
        longname: LiteralString | None = None,
        shapes: LiteralString | None = None,
        seed: spt.Seed | None = None,
    ) -> None: ...
    @override
    def __call__(
        self,
        /,
        *args: _ArrayLike_f8_co,
        loc: _ArrayLike_f8_co = 0,
        **kwds: _ArrayLike_f8_co,
    ) -> rv_discrete_frozen[Self]: ...
    @override
    def freeze(
        self,
        /,
        *args: _ArrayLike_f8_co,
        loc: _ArrayLike_f8_co = 0,
        **kwds: _ArrayLike_f8_co,
    ) -> rv_discrete_frozen[Self]: ...
    @override
    def rvs(  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        /,
        *args: _ArrayLike_f8_co,
        loc: _ArrayLike_f8_co = 0,
        size: spt.AnyInt | tuple[spt.AnyInt, ...] = 1,
        random_state: spt.Seed | None = None,
        **kwds: _ArrayLike_f8_co,
    ) -> int | _ScalarOrArray[np.int64]: ...
    @overload
    def pmf(
        self,
        /,
        k: _ScalarLike_f8_co,
        *args: _ScalarLike_f8_co,
        loc: _ScalarLike_f8_co = 0,
        **kwds: _ScalarLike_f8_co,
    ) -> np.float64: ...
    @overload
    def pmf(
        self,
        /,
        k: _ArrayLike_f8_co,
        *args: _ArrayLike_f8_co,
        loc: _ArrayLike_f8_co = 0,
        **kwds: _ArrayLike_f8_co,
    ) -> _ScalarOrArray_f8: ...
    @overload
    def logpmf(
        self,
        /,
        k: _ScalarLike_f8_co,
        *args: _ScalarLike_f8_co,
        loc: _ScalarLike_f8_co = 0,
        **kwds: _ScalarLike_f8_co,
    ) -> np.float64: ...
    @overload
    def logpmf(
        self,
        /,
        k: _ArrayLike_f8_co,
        *args: _ArrayLike_f8_co,
        loc: _ArrayLike_f8_co = 0,
        **kwds: _ArrayLike_f8_co,
    ) -> _ScalarOrArray_f8: ...
    @overload
    def cdf(
        self,
        /,
        k: _ScalarLike_f8_co,
        *args: _ScalarLike_f8_co,
        loc: _ScalarLike_f8_co = 0,
        **kwds: _ScalarLike_f8_co,
    ) -> np.float64: ...
    @overload
    def cdf(
        self,
        /,
        k: _ArrayLike_f8_co,
        *args: _ArrayLike_f8_co,
        loc: _ArrayLike_f8_co = 0,
        **kwds: _ArrayLike_f8_co,
    ) -> _ScalarOrArray_f8: ...
    @overload
    def logcdf(
        self,
        /,
        k: _ScalarLike_f8_co,
        *args: _ScalarLike_f8_co,
        loc: _ScalarLike_f8_co = 0,
        **kwds: _ScalarLike_f8_co,
    ) -> np.float64: ...
    @overload
    def logcdf(
        self,
        /,
        k: _ArrayLike_f8_co,
        *args: _ArrayLike_f8_co,
        loc: _ArrayLike_f8_co = 0,
        **kwds: _ArrayLike_f8_co,
    ) -> _ScalarOrArray_f8: ...
    @overload
    def sf(
        self,
        /,
        k: _ScalarLike_f8_co,
        *args: _ScalarLike_f8_co,
        loc: _ScalarLike_f8_co = 0,
        **kwds: _ScalarLike_f8_co,
    ) -> np.float64: ...
    @overload
    def sf(
        self,
        /,
        k: _ArrayLike_f8_co,
        *args: _ArrayLike_f8_co,
        loc: _ArrayLike_f8_co = 0,
        **kwds: _ArrayLike_f8_co,
    ) -> _ScalarOrArray_f8: ...
    @overload
    def logsf(
        self,
        /,
        k: _ScalarLike_f8_co,
        *args: _ScalarLike_f8_co,
        loc: _ScalarLike_f8_co = 0,
        **kwds: _ScalarLike_f8_co,
    ) -> np.float64: ...
    @overload
    def logsf(
        self,
        /,
        k: _ArrayLike_f8_co,
        *args: _ArrayLike_f8_co,
        loc: _ArrayLike_f8_co = 0,
        **kwds: _ArrayLike_f8_co,
    ) -> _ScalarOrArray_f8: ...
    @overload
    def ppf(
        self,
        /,
        q: _ScalarLike_f8_co,
        *args: _ScalarLike_f8_co,
        loc: _ScalarLike_f8_co = 0,
        **kwds: _ScalarLike_f8_co,
    ) -> np.float64: ...
    @overload
    def ppf(
        self,
        /,
        q: _ArrayLike_f8_co,
        *args: _ArrayLike_f8_co,
        loc: _ArrayLike_f8_co = 0,
        **kwds: _ArrayLike_f8_co,
    ) -> _ScalarOrArray_f8: ...
    @overload
    def isf(
        self,
        /,
        q: _ScalarLike_f8_co,
        *args: _ScalarLike_f8_co,
        loc: _ScalarLike_f8_co = 0,
        **kwds: _ScalarLike_f8_co,
    ) -> np.float64: ...
    @overload
    def isf(
        self,
        /,
        q: _ArrayLike_f8_co,
        *args: _ArrayLike_f8_co,
        loc: _ArrayLike_f8_co = 0,
        **kwds: _ArrayLike_f8_co,
    ) -> _ScalarOrArray_f8: ...
    def expect(
        self,
        /,
        func: Callable[[npt.NDArray[np.int_]], _Array_f8_co] | None = None,
        args: tuple[_ScalarLike_f8_co, ...] = (),
        loc: _ScalarLike_f8_co = 0,
        lb: spt.AnyInt | None = None,
        ub: spt.AnyInt | None = None,
        conditional: spt.AnyBool = False,
        maxcount: spt.AnyInt = 1000,
        tolerance: _ScalarLike_f8_co = 1e-10,
        chunksize: spt.AnyInt = 32,
    ) -> float | np.float64: ...

class rv_sample(rv_discrete, Generic[_XT_co, _PT_co]):
    xk: onpt.Array[tuple[int], _XT_co]
    pk: onpt.Array[tuple[int], _PT_co]
    qvals: onpt.Array[tuple[int], _PT_co]
    def __init__(  # pyright: ignore[reportInconsistentConstructor]
        self,
        /,
        a: _ScalarLike_f8_co = 0,
        b: _ScalarLike_f8_co = ...,
        name: LiteralString | None = None,
        badvalue: float | None = None,
        moment_tol: float = 1e-08,
        values: tuple[_ArrayLike_f8_co, _ArrayLike_f8_co] | None = None,
        inc: int = 1,
        longname: LiteralString | None = None,
        shapes: LiteralString | None = None,
        seed: spt.Seed | None = None,
    ) -> None: ...
    def _entropy(self, /) -> float | np.float64: ...
    vecentropy: Final = _entropy
    @override
    def generic_moment(self, /, n: _ArrayLikeInt_co | int | Sequence[int]) -> _Array_f8: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]

def get_distribution_names(namespace_pairs: Iterable[tuple[str, type]], rv_base_class: type) -> _Tuple2[list[LiteralString]]: ...

# private helper subtypes
@type_check_only
class _rv_continuous_0(rv_continuous):
    # overrides of rv_generic
    @override  # type: ignore[override]
    @overload
    def stats(
        self,
        /,
        loc: _ScalarLike_f8_co = 0,
        scale: _ScalarLike_f8_co = 1,
        moment: _StatsMoment = ...,
    ) -> tuple[np.float64, ...]: ...
    @overload
    def stats(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        /,
        loc: _ArrayLike_f8_co = 0,
        scale: _ArrayLike_f8_co = 1,
        moment: _StatsMoment = ...,
    ) -> tuple[np.float64, ...] | tuple[_Array_f8, ...]: ...
    @override  # type: ignore[override]
    @overload
    def entropy(self, /, loc: _ScalarLike_f8_co = 0, scale: _ScalarLike_f8_co = 1) -> np.float64: ...
    @overload
    def entropy(self, /, loc: _ArrayLike_f8_co = 0, scale: _ArrayLike_f8_co = 1) -> _ScalarOrArray_f8: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override  # type: ignore[override]
    @overload
    def moment(self, /, order: spt.AnyInt, loc: _ScalarLike_f8_co = 0, scale: _ScalarLike_f8_co = 1) -> np.float64: ...
    @overload
    def moment(self, /, order: spt.AnyInt, loc: _ArrayLike_f8_co = 0, scale: _ArrayLike_f8_co = 1) -> _ScalarOrArray_f8: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override  # type: ignore[override]
    @overload
    def median(self, /, loc: _ScalarLike_f8_co = 0, scale: _ScalarLike_f8_co = 1) -> np.float64: ...
    @overload
    def median(self, /, loc: _ArrayLike_f8_co = 0, scale: _ArrayLike_f8_co = 1) -> _ScalarOrArray_f8: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override  # type: ignore[override]
    @overload
    def mean(self, /, loc: _ScalarLike_f8_co = 0, scale: _ScalarLike_f8_co = 1) -> np.float64: ...
    @overload
    def mean(self, /, loc: _ArrayLike_f8_co = 0, scale: _ArrayLike_f8_co = 1) -> _ScalarOrArray_f8: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override  # type: ignore[override]
    @overload
    def var(self, /, loc: _ScalarLike_f8_co = 0, scale: _ScalarLike_f8_co = 1) -> np.float64: ...
    @overload
    def var(self, /, loc: _ArrayLike_f8_co = 0, scale: _ArrayLike_f8_co = 1) -> _ScalarOrArray_f8: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override  # type: ignore[override]
    @overload
    def std(self, /, loc: _ScalarLike_f8_co = 0, scale: _ScalarLike_f8_co = 1) -> np.float64: ...
    @overload
    def std(self, /, loc: _ArrayLike_f8_co = 0, scale: _ArrayLike_f8_co = 1) -> _ScalarOrArray_f8: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override  # type: ignore[override]
    @overload
    def interval(
        self,
        /,
        confidence: _ScalarLike_f8_co,
        loc: _ScalarLike_f8_co = 0,
        scale: _ScalarLike_f8_co = 1,
    ) -> _Tuple2[np.float64]: ...
    @overload
    def interval(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        /,
        confidence: _ArrayLike_f8_co,
        loc: _ArrayLike_f8_co = 0,
        scale: _ArrayLike_f8_co = 1,
    ) -> _Tuple2[np.float64] | _Tuple2[_Array_f8]: ...
    @override  # type: ignore[override]
    @overload
    def support(self, /, loc: _ScalarLike_f8_co = 0, scale: _ScalarLike_f8_co = 1) -> _Tuple2[np.float64]: ...
    @overload
    def support(self, /, loc: _ArrayLike_f8_co = 0, scale: _ArrayLike_f8_co = 1) -> _Tuple2[np.float64] | _Tuple2[_Array_f8]: ...  # pyright: ignore[reportIncompatibleMethodOverride]

    # overrides of rv_continuous
    @override
    def __call__(  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        /,
        loc: _ArrayLike_f8_co = 0,
        scale: _ArrayLike_f8_co = 1,
    ) -> rv_continuous_frozen[Self]: ...
    @override
    def freeze(  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        /,
        loc: _ArrayLike_f8_co = 0,
        scale: _ArrayLike_f8_co = 1,
    ) -> rv_continuous_frozen[Self]: ...
    @override  # type: ignore[override]
    @overload
    def pdf(self, /, x: _ScalarLike_f8_co, loc: _ScalarLike_f8_co = 0, scale: _ScalarLike_f8_co = 1) -> np.float64: ...
    @overload
    def pdf(self, /, x: _ArrayLike_f8_co, loc: _ArrayLike_f8_co = 0, scale: _ArrayLike_f8_co = 1) -> _ScalarOrArray_f8: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override  # type: ignore[override]
    @overload
    def logpdf(self, /, x: _ScalarLike_f8_co, loc: _ScalarLike_f8_co = 0, scale: _ScalarLike_f8_co = 1) -> np.float64: ...
    @overload
    def logpdf(self, /, x: _ArrayLike_f8_co, loc: _ArrayLike_f8_co = 0, scale: _ArrayLike_f8_co = 1) -> _ScalarOrArray_f8: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override  # type: ignore[override]
    @overload
    def cdf(self, /, x: _ScalarLike_f8_co, loc: _ScalarLike_f8_co = 0, scale: _ScalarLike_f8_co = 1) -> np.float64: ...
    @overload
    def cdf(self, /, x: _ArrayLike_f8_co, loc: _ArrayLike_f8_co = 0, scale: _ArrayLike_f8_co = 1) -> _ScalarOrArray_f8: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override  # type: ignore[override]
    @overload
    def logcdf(self, /, x: _ScalarLike_f8_co, loc: _ScalarLike_f8_co = 0, scale: _ScalarLike_f8_co = 1) -> np.float64: ...
    @overload
    def logcdf(self, /, x: _ArrayLike_f8_co, loc: _ArrayLike_f8_co = 0, scale: _ArrayLike_f8_co = 1) -> _ScalarOrArray_f8: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override  # type: ignore[override]
    @overload
    def sf(self, /, x: _ScalarLike_f8_co, loc: _ScalarLike_f8_co = 0, scale: _ScalarLike_f8_co = 1) -> np.float64: ...
    @overload
    def sf(self, /, x: _ArrayLike_f8_co, loc: _ArrayLike_f8_co = 0, scale: _ArrayLike_f8_co = 1) -> _ScalarOrArray_f8: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override  # type: ignore[override]
    @overload
    def logsf(self, /, x: _ScalarLike_f8_co, loc: _ScalarLike_f8_co = 0, scale: _ScalarLike_f8_co = 1) -> np.float64: ...
    @overload
    def logsf(self, /, x: _ArrayLike_f8_co, loc: _ArrayLike_f8_co = 0, scale: _ArrayLike_f8_co = 1) -> _ScalarOrArray_f8: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override  # type: ignore[override]
    @overload
    def ppf(self, /, q: _ScalarLike_f8_co, loc: _ScalarLike_f8_co = 0, scale: _ScalarLike_f8_co = 1) -> np.float64: ...
    @overload
    def ppf(self, /, q: _ArrayLike_f8_co, loc: _ArrayLike_f8_co = 0, scale: _ArrayLike_f8_co = 1) -> _ScalarOrArray_f8: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override  # type: ignore[override]
    @overload
    def isf(self, /, q: _ScalarLike_f8_co, loc: _ScalarLike_f8_co = 0, scale: _ScalarLike_f8_co = 1) -> np.float64: ...
    @overload
    def isf(self, /, q: _ArrayLike_f8_co, loc: _ArrayLike_f8_co = 0, scale: _ArrayLike_f8_co = 1) -> _ScalarOrArray_f8: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def rvs(  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        /,
        loc: _ScalarLike_f8_co = 0,
        scale: _ScalarLike_f8_co = 1,
        size: spt.AnyInt | tuple[spt.AnyInt, ...] = 1,
        random_state: spt.Seed | None = None,
    ) -> _ScalarOrArray_f8: ...
