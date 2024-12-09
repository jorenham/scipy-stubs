# NOTE: Using `@override` on `__call__` or `freeze` in `rv_discrete` causes stubtest to crash (mypy 1.11.1 and 1.13.0)
# mypy: disable-error-code="explicit-override, override"

import abc
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Final, Generic, Literal, TypeAlias, overload, type_check_only
from typing_extensions import LiteralString, Self, TypeVar, Unpack, override

import numpy as np
import optype as op
import optype.numpy as onp
import scipy._typing as spt
from scipy.integrate._typing import QuadOpts as _QuadOpts

_T = TypeVar("_T")
_Tuple2: TypeAlias = tuple[_T, _T]
_Tuple3: TypeAlias = tuple[_T, _T, _T]
_Tuple4: TypeAlias = tuple[_T, _T, _T, _T]

_Scalar_i: TypeAlias = np.integer[Any]
_Scalar_f: TypeAlias = np.float64 | np.float32 | np.float16  # longdouble often results in trouble
_Scalar_if: TypeAlias = _Scalar_f | _Scalar_i  # including np.bool_ here would become messy

# NOTE: this will be equivalent to `float` in `numpy>=2.2`, see https://github.com/numpy/numpy/pull/27334
_Scalar_b1: TypeAlias = bool | np.bool_
_Scalar_i8: TypeAlias = int | np.int64
_Scalar_f8: TypeAlias = float | np.float64

_ShapeT = TypeVar("_ShapeT", bound=tuple[int, ...], default=tuple[int, ...])
_Arr_b1: TypeAlias = onp.Array[_ShapeT, np.bool_]
_Arr_i8: TypeAlias = onp.Array[_ShapeT, np.int64]
_Arr_f8: TypeAlias = onp.Array[_ShapeT, np.float64]

_ArrLike_b1: TypeAlias = _Scalar_b1 | _Arr_b1
_ArrLike_i8: TypeAlias = _Scalar_i8 | _Arr_i8
_ArrLike_f8: TypeAlias = _Scalar_f8 | _Arr_f8

_Scalar_f8_co: TypeAlias = float | _Scalar_if
_Arr_f8_co: TypeAlias = onp.Array[_ShapeT, _Scalar_if]
_ArrLike_f8_co: TypeAlias = _ArrLike_f8 | onp.CanArray[tuple[int, ...], np.dtype[_Scalar_if]] | Sequence[_ArrLike_f8_co]

_ArgT = TypeVar("_ArgT", bound=_ArrLike_f8_co, default=_ArrLike_f8_co)
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
_RVKwds: TypeAlias = dict[str, _ArrLike_f8_co]

_Moments1: TypeAlias = Literal["m", "v", "s", "k"]
_Moments2: TypeAlias = Literal[
    "mv", "ms", "mk",
    "vm", "vs", "vk",
    "sm", "sv", "sk",
    "km", "kv", "ks",
]  # fmt: skip
_Moments3: TypeAlias = Literal[
    "mvs", "mvk", "msv", "msk", "mkv", "mks",
    "vms", "vmk", "vsm", "vsk", "vkm", "vks",
    "smv", "smk", "svm", "svk", "skm", "skv",
    "kmv", "kms", "kvm", "kvs", "ksm", "ksv",
]  # fmt: skip
_Moments4: TypeAlias = Literal[
    "mvsk", "mvks", "msvk", "mskv", "mkvs", "mksv",
    "vmsk", "vmks", "vsmk", "vskm", "vkms", "vksm",
    "smvk", "smkv", "svmk", "svkm", "skmv", "skvm",
    "kmvs", "kmsv", "kvms", "kvsm", "ksmv", "ksvm",
]  # fmt: skip

_FitMethod: TypeAlias = Literal["MLE", "MM"]

###

docheaders: Final[dict[str, str]] = ...
docdict: Final[dict[str, str]] = ...
docdict_discrete: Final[dict[str, str]] = ...
parse_arg_template: Final[str] = ...

def argsreduce(cond: _Arr_b1, *args: _ArrLike_f8_co) -> list[_Arr_f8_co]: ...

_RVT = TypeVar("_RVT", bound=rv_generic, default=rv_generic)
_RVT_co = TypeVar("_RVT_co", bound=rv_generic, covariant=True, default=rv_generic)
_VT_f8 = TypeVar("_VT_f8", bound=_ArrLike_f8, default=_ArrLike_f8)
_VT_f8_co = TypeVar("_VT_f8_co", bound=_ArrLike_f8, covariant=True, default=_ArrLike_f8)

class rv_frozen(Generic[_RVT_co, _VT_f8_co]):
    dist: _RVT_co
    args: _RVArgs[_VT_f8_co]
    kwds: _RVKwds

    @property
    def random_state(self, /) -> spt.RNG: ...
    @random_state.setter
    def random_state(self, seed: spt.Seed, /) -> None: ...

    #
    @overload
    def __init__(self: rv_frozen[_RVT, _Scalar_f8], /, dist: _RVT) -> None: ...
    @overload
    def __init__(self, /, dist: _RVT_co, *args: _VT_f8_co, **kwds: _VT_f8_co) -> None: ...
    @overload
    def __init__(self, /, dist: _RVT_co, *args: _ArrLike_f8_co, **kwds: _ArrLike_f8_co) -> None: ...

    #
    @overload
    def cdf(self, /, x: _Scalar_f8_co) -> _VT_f8_co: ...
    @overload
    def cdf(self, /, x: _Arr_f8_co[Any]) -> _Arr_f8: ...
    @overload
    def cdf(self, /, x: _ArrLike_f8_co) -> _VT_f8_co | _Arr_f8: ...
    #
    @overload
    def logcdf(self, /, x: _Scalar_f8_co) -> _VT_f8_co: ...
    @overload
    def logcdf(self, /, x: _Arr_f8_co[Any]) -> _Arr_f8: ...
    @overload
    def logcdf(self, /, x: _ArrLike_f8_co) -> _VT_f8_co | _Arr_f8: ...

    #
    @overload
    def sf(self, /, x: _Scalar_f8_co) -> _VT_f8_co: ...
    @overload
    def sf(self, /, x: _Arr_f8_co[Any]) -> _Arr_f8: ...
    @overload
    def sf(self, /, x: _ArrLike_f8_co) -> _VT_f8_co | _Arr_f8: ...
    #
    @overload
    def logsf(self, /, x: _Scalar_f8_co) -> _VT_f8_co: ...
    @overload
    def logsf(self, /, x: _Arr_f8_co[Any]) -> _Arr_f8: ...
    @overload
    def logsf(self, /, x: _ArrLike_f8_co) -> _VT_f8_co | _Arr_f8: ...

    #
    @overload
    def ppf(self, /, q: _Scalar_f8_co) -> _VT_f8_co: ...
    @overload
    def ppf(self, /, q: _Arr_f8_co[Any]) -> _Arr_f8: ...
    @overload
    def ppf(self, /, q: _ArrLike_f8_co) -> _VT_f8_co | _Arr_f8: ...
    #
    @overload
    def isf(self, /, q: _Scalar_f8_co) -> _VT_f8_co: ...
    @overload
    def isf(self, /, q: _Arr_f8_co[Any]) -> _Arr_f8: ...
    @overload
    def isf(self, /, q: _ArrLike_f8_co) -> _VT_f8_co | _Arr_f8: ...

    #
    def rvs(self, /, size: spt.AnyShape | None = None, random_state: spt.Seed | None = None) -> _ArrLike_f8: ...

    #
    @overload
    def stats(self, /, moments: _Moments1) -> _VT_f8_co: ...
    @overload
    def stats(self, /, moments: _Moments2 = ...) -> _Tuple2[_VT_f8_co]: ...
    @overload
    def stats(self, /, moments: _Moments3) -> _Tuple3[_VT_f8_co]: ...
    @overload
    def stats(self, /, moments: _Moments4) -> _Tuple4[_VT_f8_co]: ...
    #
    def median(self, /) -> _VT_f8_co: ...
    def mean(self, /) -> _VT_f8_co: ...
    def var(self, /) -> _VT_f8_co: ...
    def std(self, /) -> _VT_f8_co: ...
    # order defaults to `None`, but that will `raise TypeError`
    def moment(self, /, order: int | _Scalar_i | None = None) -> _VT_f8_co: ...
    def entropy(self, /) -> _VT_f8_co: ...
    #
    def interval(self, /, confidence: _Scalar_f8_co | None = None) -> _Tuple2[_VT_f8_co]: ...
    def support(self, /) -> _Tuple2[_VT_f8_co]: ...

    #
    def expect(
        self: rv_frozen[_RVT, _Scalar_f8],
        /,
        func: Callable[[float], _Scalar_f8_co] | None = None,
        lb: _Scalar_f8_co | None = None,
        ub: _Scalar_f8_co | None = None,
        conditional: _Scalar_b1 = False,
        **kwds: Unpack[_QuadOpts],
    ) -> _Scalar_f8: ...

_RVT_c_co = TypeVar("_RVT_c_co", bound=rv_continuous, covariant=True, default=rv_continuous)

class rv_continuous_frozen(rv_frozen[_RVT_c_co, _VT_f8_co], Generic[_RVT_c_co, _VT_f8_co]):
    @overload
    def pdf(self, /, x: _Scalar_f8_co) -> _VT_f8_co: ...
    @overload
    def pdf(self, /, x: _Arr_f8_co[Any]) -> _Arr_f8: ...
    @overload
    def pdf(self, /, x: _ArrLike_f8_co) -> _VT_f8_co | _Arr_f8: ...
    #
    @overload
    def logpdf(self, /, x: _Scalar_f8_co) -> _VT_f8_co: ...
    @overload
    def logpdf(self, /, x: _Arr_f8_co[Any]) -> _Arr_f8: ...
    @overload
    def logpdf(self, /, x: _ArrLike_f8_co) -> _VT_f8_co | _Arr_f8: ...

_RVT_d_co = TypeVar("_RVT_d_co", bound=rv_discrete, covariant=True, default=rv_discrete)

class rv_discrete_frozen(rv_frozen[_RVT_d_co, _VT_f8_co], Generic[_RVT_d_co, _VT_f8_co]):
    @overload
    def pmf(self, /, k: _Scalar_f8_co) -> _VT_f8_co: ...
    @overload
    def pmf(self, /, k: _Arr_f8_co[Any]) -> _Arr_f8: ...
    @overload
    def pmf(self, /, k: _ArrLike_f8_co) -> _VT_f8_co | _Arr_f8: ...
    #
    @overload
    def logpmf(self, /, k: _Scalar_f8_co) -> _VT_f8_co: ...
    @overload
    def logpmf(self, /, k: _Arr_f8_co[Any]) -> _Arr_f8: ...
    @overload
    def logpmf(self, /, k: _ArrLike_f8_co) -> _VT_f8_co | _Arr_f8: ...

# NOTE: Because of the limitations of `ParamSpec`, there is no proper way to annotate specific "positional or keyword arguments".
# Considering the Liskov Substitution Principle, the only remaining option is to annotate `*args, and `**kwargs` as `Any`.
class rv_generic:
    def __init__(self, /, seed: spt.Seed | None = None) -> None: ...
    @property
    def random_state(self, /) -> spt.RNG: ...
    @random_state.setter
    def random_state(self, seed: spt.Seed, /) -> None: ...
    @abc.abstractmethod
    def _attach_methods(self, /) -> None: ...
    def _attach_argparser_methods(self, /) -> None: ...
    def _construct_argparser(
        self,
        /,
        meths_to_inspect: Iterable[Callable[..., Any]],
        locscale_in: str,
        locscale_out: str,
    ) -> None: ...
    def _construct_doc(self, /, docdict: dict[str, str], shapes_vals: tuple[float, ...] | None = None) -> None: ...
    def _construct_default_doc(
        self,
        /,
        longname: str | None = None,
        docdict: dict[str, str] | None = None,
        discrete: Literal["continuous", "discrete"] = "continuous",
    ) -> None: ...
    @overload
    def __call__(self, /) -> rv_frozen[Self, _Scalar_f8]: ...
    @overload
    def __call__(self, /, *args: _Scalar_f8_co, **kwds: _Scalar_f8_co) -> rv_frozen[Self, _Scalar_f8]: ...
    @overload
    def __call__(self, /, *args: _ArrLike_f8_co, **kwds: _ArrLike_f8_co) -> rv_frozen[Self]: ...
    @overload
    def freeze(self, /) -> rv_frozen[Self, _Scalar_f8]: ...
    @overload
    def freeze(self, /, *args: _Scalar_f8_co, **kwds: _Scalar_f8_co) -> rv_frozen[Self, _Scalar_f8]: ...
    @overload
    def freeze(self, /, *args: _ArrLike_f8_co, **kwds: _ArrLike_f8_co) -> rv_frozen[Self]: ...
    def _stats(self, /, *args: Any, **kwds: Any) -> _Tuple4[_Scalar_f8 | None] | _Tuple4[_Arr_f8 | None]: ...
    def _munp(self, /, n: onp.ToInt | onp.ToIntND, *args: Any) -> _Arr_f8: ...
    def _argcheck_rvs(
        self,
        /,
        *args: Any,
        size: onp.ToInt | onp.ToIntND | None = None,
    ) -> tuple[list[_Arr_f8_co], _Arr_f8_co, _Arr_f8_co, tuple[int, ...] | tuple[np.int_, ...]]: ...
    def _argcheck(self, /, *args: Any) -> _ArrLike_b1: ...
    def _get_support(self, /, *args: Any, **kwargs: Any) -> _Tuple2[_ArrLike_f8]: ...
    def _support_mask(self, /, x: _Arr_f8_co, *args: Any) -> _Arr_b1: ...
    def _open_support_mask(self, /, x: _Arr_f8_co, *args: Any) -> _ArrLike_b1: ...
    def _rvs(self, /, *args: Any, size: spt.AnyShape | None = None, random_state: spt.Seed | None = None) -> _ArrLike_f8: ...
    def _logcdf(self, /, x: _VT_f8, *args: Any) -> _VT_f8: ...
    def _sf(self, /, x: _VT_f8, *args: Any) -> _VT_f8: ...
    def _logsf(self, /, x: _VT_f8, *args: Any) -> _VT_f8: ...
    def _ppf(self, /, q: _VT_f8, *args: Any) -> _VT_f8: ...
    def _isf(self, /, q: _VT_f8, *args: Any) -> _VT_f8: ...
    @overload
    def rvs(
        self,
        /,
        *args: _Scalar_f8_co,
        random_state: spt.Seed,
        discrete: Literal[True, 1],
        **kwds: _ArrLike_f8_co,
    ) -> _ArrLike_i8: ...
    @overload
    def rvs(
        self,
        /,
        *args: _Scalar_f8_co,
        random_state: spt.Seed,
        discrete: Literal[False, 0] | None = ...,
        **kwds: _ArrLike_f8_co,
    ) -> _ArrLike_f8: ...
    @overload
    def stats(self, /, *args: _Scalar_f8_co, moment: _Moments1, **kwds: _Scalar_f8_co) -> _Scalar_f8: ...
    @overload
    def stats(self, /, *args: _ArrLike_f8_co, moment: _Moments1, **kwds: _ArrLike_f8_co) -> _Scalar_f8 | _Arr_f8: ...
    @overload
    def stats(self, /, *args: _Scalar_f8_co, moment: _Moments2 = ..., **kwds: _Scalar_f8_co) -> _Tuple2[_Scalar_f8]: ...
    @overload
    def stats(self, /, *args: _ArrLike_f8_co, moment: _Moments2 = ..., **kwds: _ArrLike_f8_co) -> _Tuple2[_ArrLike_f8]: ...
    @overload
    def stats(self, /, *args: _Scalar_f8_co, moment: _Moments3, **kwds: _Scalar_f8_co) -> _Tuple3[_Scalar_f8]: ...
    @overload
    def stats(self, /, *args: _ArrLike_f8_co, moment: _Moments3, **kwds: _ArrLike_f8_co) -> _Tuple3[_ArrLike_f8]: ...
    @overload
    def stats(self, /, *args: _Scalar_f8_co, moment: _Moments4, **kwds: _Scalar_f8_co) -> _Tuple4[_Scalar_f8]: ...
    @overload
    def stats(self, /, *args: _ArrLike_f8_co, moment: _Moments4, **kwds: _ArrLike_f8_co) -> _Tuple4[_ArrLike_f8]: ...
    @overload
    def entropy(self, /, *args: _Scalar_f8_co, **kwds: _Scalar_f8_co) -> _Scalar_f8: ...
    @overload
    def entropy(self, /, *args: _ArrLike_f8_co, **kwds: _ArrLike_f8_co) -> _ArrLike_f8: ...
    @overload
    def moment(self, /, order: onp.ToInt, *args: _Scalar_f8_co, **kwds: _Scalar_f8_co) -> _Scalar_f8: ...
    @overload
    def moment(self, /, order: onp.ToInt, *args: _ArrLike_f8_co, **kwds: _ArrLike_f8_co) -> _ArrLike_f8: ...
    @overload
    def median(self, /, *args: _Scalar_f8_co, **kwds: _Scalar_f8_co) -> _Scalar_f8: ...
    @overload
    def median(self, /, *args: _ArrLike_f8_co, **kwds: _ArrLike_f8_co) -> _ArrLike_f8: ...
    @overload
    def mean(self, /, *args: _Scalar_f8_co, **kwds: _Scalar_f8_co) -> _Scalar_f8: ...
    @overload
    def mean(self, /, *args: _ArrLike_f8_co, **kwds: _ArrLike_f8_co) -> _ArrLike_f8: ...
    @overload
    def var(self, /, *args: _Scalar_f8_co, **kwds: _Scalar_f8_co) -> _Scalar_f8: ...
    @overload
    def var(self, /, *args: _ArrLike_f8_co, **kwds: _ArrLike_f8_co) -> _ArrLike_f8: ...
    @overload
    def std(self, /, *args: _Scalar_f8_co, **kwds: _Scalar_f8_co) -> _Scalar_f8: ...
    @overload
    def std(self, /, *args: _ArrLike_f8_co, **kwds: _ArrLike_f8_co) -> _ArrLike_f8: ...
    @overload
    def interval(self, /, confidence: _Scalar_f8_co, *args: _Scalar_f8_co, **kwds: _Scalar_f8_co) -> _Tuple2[_Scalar_f8]: ...
    @overload
    def interval(self, /, confidence: _ArrLike_f8_co, *args: _ArrLike_f8_co, **kwds: _ArrLike_f8_co) -> _Tuple2[_ArrLike_f8]: ...
    @overload
    def support(self, /, *args: _Scalar_f8_co, **kwds: _Scalar_f8_co) -> _Tuple2[_Scalar_f8]: ...
    @overload
    def support(self, /, *args: _ArrLike_f8_co, **kwds: _ArrLike_f8_co) -> _Tuple2[_Scalar_f8] | _Tuple2[_Arr_f8]: ...
    def nnlf(self, /, theta: Sequence[_Scalar_f8_co], x: _ArrLike_f8_co) -> _ArrLike_f8: ...
    def _nnlf(self, /, x: _Arr_f8_co, *args: Any) -> _ArrLike_f8: ...
    def _penalized_nnlf(self, /, theta: Sequence[Any], x: _Arr_f8_co) -> _Scalar_f8: ...
    def _penalized_nlpsf(self, /, theta: Sequence[Any], x: _Arr_f8_co) -> _Scalar_f8: ...

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

    def _shape_info(self, /) -> list[_ShapeInfo]: ...
    def _param_info(self, /) -> list[_ShapeInfo]: ...
    def _attach_methods(self, /) -> None: ...
    def generic_moment(self, /, n: onp.ToInt | onp.ToIntND, *args: _Scalar_f8_co) -> _Arr_f8: ...
    def _logpxf(self, /, x: _Arr_f8_co, *args: Any) -> _Arr_f8: ...
    def _cdf_single(self, /, x: _Scalar_f8_co, *args: Any) -> _Scalar_f8: ...
    def _cdfvec(self, /, x: _VT_f8, *args: Any) -> _VT_f8: ...
    def _cdf(self, /, x: _VT_f8, *args: Any) -> _VT_f8: ...
    def _ppfvec(self, /, q: _VT_f8, *args: Any) -> _VT_f8: ...
    def _unpack_loc_scale(
        self,
        /,
        theta: Sequence[_ArrLike_f8_co],
    ) -> tuple[_ArrLike_f8_co, _ArrLike_f8_co, tuple[_ArrLike_f8_co, ...]]: ...

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
    @overload
    def __call__(self, /) -> rv_continuous_frozen[Self, _Scalar_f8]: ...
    @overload
    def __call__(
        self,
        /,
        *args: _Scalar_f8_co,
        loc: _Scalar_f8_co = 0,
        scale: _Scalar_f8_co = 1,
        **kwds: _Scalar_f8_co,
    ) -> rv_continuous_frozen[Self, _Scalar_f8]: ...
    @overload
    def __call__(
        self,
        /,
        *args: _ArrLike_f8_co,
        loc: _ArrLike_f8_co = 0,
        scale: _ArrLike_f8_co = 1,
        **kwds: _ArrLike_f8_co,
    ) -> rv_continuous_frozen[Self]: ...
    #
    @overload
    def freeze(self, /) -> rv_continuous_frozen[Self, _Scalar_f8]: ...
    @overload
    def freeze(
        self,
        /,
        *args: _Scalar_f8_co,
        loc: _Scalar_f8_co = 0,
        scale: _Scalar_f8_co = 1,
        **kwds: _Scalar_f8_co,
    ) -> rv_continuous_frozen[Self, _Scalar_f8]: ...
    @overload
    def freeze(
        self,
        /,
        *args: _ArrLike_f8_co,
        loc: _ArrLike_f8_co = 0,
        scale: _ArrLike_f8_co = 1,
        **kwds: _ArrLike_f8_co,
    ) -> rv_continuous_frozen[Self]: ...

    #
    def _pdf(self, /, x: _VT_f8, *args: Any) -> _VT_f8: ...
    def _logpdf(self, /, x: _VT_f8, *args: Any) -> _VT_f8: ...
    #
    @overload
    def pdf(
        self,
        /,
        x: _Scalar_f8_co,
        *args: _Scalar_f8_co,
        loc: _Scalar_f8_co = 0,
        scale: _Scalar_f8_co = 1,
        **kwds: _Scalar_f8_co,
    ) -> _Scalar_f8: ...
    @overload
    def pdf(
        self,
        /,
        x: _Arr_f8_co[_ShapeT],
        *args: _Scalar_f8_co,
        loc: _Scalar_f8_co = 0,
        scale: _Scalar_f8_co = 1,
        **kwds: _Scalar_f8_co,
    ) -> _Arr_f8[_ShapeT]: ...
    @overload
    def pdf(
        self,
        /,
        x: _ArrLike_f8_co,
        *args: _ArrLike_f8_co,
        loc: _ArrLike_f8_co = 0,
        scale: _ArrLike_f8_co = 1,
        **kwds: _ArrLike_f8_co,
    ) -> _ArrLike_f8: ...
    #
    @overload
    def logpdf(
        self,
        /,
        x: _Scalar_f8_co,
        *args: _Scalar_f8_co,
        loc: _Scalar_f8_co = 0,
        scale: _Scalar_f8_co = 1,
        **kwds: _Scalar_f8_co,
    ) -> _Scalar_f8: ...
    @overload
    def logpdf(
        self,
        /,
        x: _Arr_f8_co[_ShapeT],
        *args: _Scalar_f8_co,
        loc: _Scalar_f8_co = 0,
        scale: _Scalar_f8_co = 1,
        **kwds: _Scalar_f8_co,
    ) -> _Arr_f8[_ShapeT]: ...
    @overload
    def logpdf(
        self,
        /,
        x: _ArrLike_f8_co,
        *args: _ArrLike_f8_co,
        loc: _ArrLike_f8_co = 0,
        scale: _ArrLike_f8_co = 1,
        **kwds: _ArrLike_f8_co,
    ) -> _ArrLike_f8: ...

    #
    @overload
    def cdf(
        self,
        /,
        x: _Scalar_f8_co,
        *args: _Scalar_f8_co,
        loc: _Scalar_f8_co = 0,
        scale: _Scalar_f8_co = 1,
        **kwds: _Scalar_f8_co,
    ) -> _Scalar_f8: ...
    @overload
    def cdf(
        self,
        /,
        x: _Arr_f8_co[_ShapeT],
        *args: _Scalar_f8_co,
        loc: _Scalar_f8_co = 0,
        scale: _Scalar_f8_co = 1,
        **kwds: _Scalar_f8_co,
    ) -> _Arr_f8[_ShapeT]: ...
    @overload
    def cdf(
        self,
        /,
        x: _ArrLike_f8_co,
        *args: _ArrLike_f8_co,
        loc: _ArrLike_f8_co = 0,
        scale: _ArrLike_f8_co = 1,
        **kwds: _ArrLike_f8_co,
    ) -> _ArrLike_f8: ...
    #
    @overload
    def logcdf(
        self,
        /,
        x: _Scalar_f8_co,
        *args: _Scalar_f8_co,
        loc: _Scalar_f8_co = 0,
        scale: _Scalar_f8_co = 1,
        **kwds: _Scalar_f8_co,
    ) -> _Scalar_f8: ...
    @overload
    def logcdf(
        self,
        /,
        x: _Arr_f8_co[_ShapeT],
        *args: _Scalar_f8_co,
        loc: _Scalar_f8_co = 0,
        scale: _Scalar_f8_co = 1,
        **kwds: _Scalar_f8_co,
    ) -> _Arr_f8[_ShapeT]: ...
    @overload
    def logcdf(
        self,
        /,
        x: _ArrLike_f8_co,
        *args: _ArrLike_f8_co,
        loc: _ArrLike_f8_co = 0,
        scale: _ArrLike_f8_co = 1,
        **kwds: _ArrLike_f8_co,
    ) -> _ArrLike_f8: ...

    #
    @overload
    def sf(
        self,
        /,
        x: _Scalar_f8_co,
        *args: _Scalar_f8_co,
        loc: _Scalar_f8_co = 0,
        scale: _Scalar_f8_co = 1,
        **kwds: _Scalar_f8_co,
    ) -> _Scalar_f8: ...
    @overload
    def sf(
        self,
        /,
        x: _Arr_f8_co[_ShapeT],
        *args: _Scalar_f8_co,
        loc: _Scalar_f8_co = 0,
        scale: _Scalar_f8_co = 1,
        **kwds: _Scalar_f8_co,
    ) -> _Arr_f8[_ShapeT]: ...
    @overload
    def sf(
        self,
        /,
        x: _ArrLike_f8_co,
        *args: _ArrLike_f8_co,
        loc: _ArrLike_f8_co = 0,
        scale: _ArrLike_f8_co = 1,
        **kwds: _ArrLike_f8_co,
    ) -> _ArrLike_f8: ...
    #
    @overload
    def logsf(
        self,
        /,
        x: _Scalar_f8_co,
        *args: _Scalar_f8_co,
        loc: _Scalar_f8_co = 0,
        scale: _Scalar_f8_co = 1,
        **kwds: _Scalar_f8_co,
    ) -> _Scalar_f8: ...
    @overload
    def logsf(
        self,
        /,
        x: _Arr_f8_co[_ShapeT],
        *args: _Scalar_f8_co,
        loc: _Scalar_f8_co = 0,
        scale: _Scalar_f8_co = 1,
        **kwds: _Scalar_f8_co,
    ) -> _Arr_f8[_ShapeT]: ...
    @overload
    def logsf(
        self,
        /,
        x: _ArrLike_f8_co,
        *args: _ArrLike_f8_co,
        loc: _ArrLike_f8_co = 0,
        scale: _ArrLike_f8_co = 1,
        **kwds: _ArrLike_f8_co,
    ) -> _ArrLike_f8: ...

    #
    @overload
    def ppf(
        self,
        /,
        q: _Scalar_f8_co,
        *args: _Scalar_f8_co,
        loc: _Scalar_f8_co = 0,
        scale: _Scalar_f8_co = 1,
        **kwds: _Scalar_f8_co,
    ) -> _Scalar_f8: ...
    @overload
    def ppf(
        self,
        /,
        q: _Arr_f8_co[_ShapeT],
        *args: _Scalar_f8_co,
        loc: _Scalar_f8_co = 0,
        scale: _Scalar_f8_co = 1,
        **kwds: _Scalar_f8_co,
    ) -> _Arr_f8[_ShapeT]: ...
    @overload
    def ppf(
        self,
        /,
        q: _ArrLike_f8_co,
        *args: _ArrLike_f8_co,
        loc: _ArrLike_f8_co = 0,
        scale: _ArrLike_f8_co = 1,
        **kwds: _ArrLike_f8_co,
    ) -> _ArrLike_f8: ...
    #
    @overload
    def isf(
        self,
        /,
        q: _Scalar_f8_co,
        *args: _Scalar_f8_co,
        loc: _Scalar_f8_co = 0,
        scale: _Scalar_f8_co = 1,
        **kwds: _Scalar_f8_co,
    ) -> _Scalar_f8: ...
    @overload
    def isf(
        self,
        /,
        q: _Arr_f8_co[_ShapeT],
        *args: _Scalar_f8_co,
        loc: _Scalar_f8_co = 0,
        scale: _Scalar_f8_co = 1,
        **kwds: _Scalar_f8_co,
    ) -> _Arr_f8[_ShapeT]: ...
    @overload
    def isf(
        self,
        /,
        q: _ArrLike_f8_co,
        *args: _ArrLike_f8_co,
        loc: _ArrLike_f8_co = 0,
        scale: _ArrLike_f8_co = 1,
        **kwds: _ArrLike_f8_co,
    ) -> _ArrLike_f8: ...

    #
    def _nnlf_and_penalty(self, /, x: _Arr_f8, args: Sequence[Any]) -> _Scalar_f8: ...
    def _fitstart(
        self,
        /,
        data: _Arr_f8,
        args: tuple[Any, ...] | None = None,
    ) -> tuple[Unpack[tuple[_Scalar_f8, ...]], _Scalar_f8, _Scalar_f8]: ...
    def _reduce_func(
        self,
        /,
        args: tuple[Any, ...],
        kwds: dict[str, Any],
        data: _ArrLike_f8_co | None = None,
    ) -> tuple[
        list[_Scalar_f8],
        Callable[[list[_Scalar_f8_co], _Arr_f8_co], _Scalar_f8],
        Callable[[list[_Scalar_f8_co], _Arr_f8_co], list[_Scalar_f8]],
        list[_Scalar_f8],
    ]: ...
    def _moment_error(self, /, theta: list[_Scalar_f8_co], x: _Arr_f8_co, data_moments: _Arr_f8_co[tuple[int]]) -> _Scalar_f8: ...
    def _fit_loc_scale_support(self, /, data: _ArrLike_f8_co, *args: Any) -> _Tuple2[np.intp] | _Tuple2[_Scalar_f8]: ...
    def fit_loc_scale(self, /, data: _ArrLike_f8_co, *args: _Scalar_f8_co) -> _Tuple2[_Scalar_f8]: ...
    def fit(
        self,
        /,
        data: _ArrLike_f8_co,
        *args: _Scalar_f8_co,
        optimizer: Callable[[_Arr_f8, tuple[_Scalar_f8, ...], tuple[_Scalar_f8, ...], bool], tuple[_Scalar_f8, ...]],
        method: _FitMethod = "MLE",
        **kwds: _Scalar_f8_co,
    ) -> tuple[_Scalar_f8, ...]: ...

    #
    def expect(
        self,
        /,
        func: Callable[[float], _Scalar_f8] | None = None,
        args: tuple[_Scalar_f8_co, ...] = (),
        loc: _Scalar_f8_co = 0,
        scale: _Scalar_f8_co = 1,
        lb: _Scalar_f8_co | None = None,
        ub: _Scalar_f8_co | None = None,
        conditional: op.CanBool = False,
        **kwds: Unpack[_QuadOpts],
    ) -> _Scalar_f8: ...

    #
    @override
    def rvs(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        /,
        *args: _Scalar_f8_co,
        loc: _Scalar_f8_co = 0,
        scale: _Scalar_f8_co = 1,
        size: spt.AnyShape = 1,
        random_state: spt.Seed | None = None,
        **kwds: _ArrLike_f8_co,
    ) -> _ArrLike_f8: ...

class rv_discrete(_rv_mixin, rv_generic):
    inc: Final[int]
    moment_tol: Final[float]

    def __new__(
        cls,
        a: _Scalar_f8_co = 0,
        b: _Scalar_f8_co = ...,
        name: LiteralString | None = None,
        badvalue: _Scalar_f8 | None = None,
        moment_tol: _Scalar_f8 = 1e-08,
        values: _Tuple2[_ArrLike_f8_co] | None = None,
        inc: int | np.int_ = 1,
        longname: LiteralString | None = None,
        shapes: LiteralString | None = None,
        seed: spt.Seed | None = None,
    ) -> Self: ...
    def __init__(  # pyright: ignore[reportInconsistentConstructor]
        self,
        /,
        a: _Scalar_f8_co = 0,
        b: _Scalar_f8_co = ...,
        name: LiteralString | None = None,
        badvalue: _Scalar_f8 | None = None,
        moment_tol: _Scalar_f8 = 1e-08,
        values: None = None,
        inc: int | np.int_ = 1,
        longname: LiteralString | None = None,
        shapes: LiteralString | None = None,
        seed: spt.Seed | None = None,
    ) -> None: ...

    #
    # NOTE: Using `@override` on `__call__` or `freeze` causes stubtest to crash (mypy 1.11.1)
    @overload
    def __call__(self, /) -> rv_discrete_frozen[Self, _Scalar_f8]: ...
    @overload
    def __call__(
        self,
        /,
        *args: _Scalar_f8_co,
        loc: _Scalar_f8_co = 0,
        **kwds: _Scalar_f8_co,
    ) -> rv_discrete_frozen[Self, _Scalar_f8]: ...
    @overload
    def __call__(self, /, *args: _ArrLike_f8_co, loc: _ArrLike_f8_co = 0, **kwds: _ArrLike_f8_co) -> rv_discrete_frozen[Self]: ...
    #
    @overload
    def freeze(self, /) -> rv_discrete_frozen[Self, _Scalar_f8]: ...
    @overload
    def freeze(
        self,
        /,
        *args: _Scalar_f8_co,
        loc: _Scalar_f8_co = 0,
        **kwds: _Scalar_f8_co,
    ) -> rv_discrete_frozen[Self, _Scalar_f8]: ...
    @overload
    def freeze(self, /, *args: _ArrLike_f8_co, loc: _ArrLike_f8_co = 0, **kwds: _ArrLike_f8_co) -> rv_discrete_frozen[Self]: ...

    #
    @overload
    def pmf(self, /, k: _Scalar_f8_co, *args: _Scalar_f8_co, loc: _Scalar_f8_co = 0, **kwds: _Scalar_f8_co) -> _Scalar_f8: ...
    @overload
    def pmf(
        self,
        /,
        k: _Arr_f8_co[_ShapeT],
        *args: _Scalar_f8_co,
        loc: _Scalar_f8_co = 0,
        **kwds: _Scalar_f8_co,
    ) -> _Arr_f8[_ShapeT]: ...
    @overload
    def pmf(
        self,
        /,
        k: _ArrLike_f8_co,
        *args: _ArrLike_f8_co,
        loc: _ArrLike_f8_co = 0,
        **kwds: _ArrLike_f8_co,
    ) -> _ArrLike_f8: ...
    #
    @overload
    def logpmf(self, /, k: _Scalar_f8_co, *args: _Scalar_f8_co, loc: _Scalar_f8_co = 0, **kwds: _Scalar_f8_co) -> _Scalar_f8: ...
    @overload
    def logpmf(
        self,
        /,
        k: _Arr_f8_co[_ShapeT],
        *args: _Scalar_f8_co,
        loc: _Scalar_f8_co = 0,
        **kwds: _Scalar_f8_co,
    ) -> _Arr_f8[_ShapeT]: ...
    @overload
    def logpmf(
        self,
        /,
        k: _ArrLike_f8_co,
        *args: _ArrLike_f8_co,
        loc: _ArrLike_f8_co = 0,
        **kwds: _ArrLike_f8_co,
    ) -> _ArrLike_f8: ...

    #
    @overload
    def cdf(self, /, k: _Scalar_f8_co, *args: _Scalar_f8_co, loc: _Scalar_f8_co = 0, **kwds: _Scalar_f8_co) -> _Scalar_f8: ...
    @overload
    def cdf(
        self,
        /,
        k: _Arr_f8_co[_ShapeT],
        *args: _Scalar_f8_co,
        loc: _Scalar_f8_co = 0,
        **kwds: _Scalar_f8_co,
    ) -> _Arr_f8[_ShapeT]: ...
    @overload
    def cdf(
        self,
        /,
        k: _ArrLike_f8_co,
        *args: _ArrLike_f8_co,
        loc: _ArrLike_f8_co = 0,
        **kwds: _ArrLike_f8_co,
    ) -> _ArrLike_f8: ...
    #
    @overload
    def logcdf(self, /, k: _Scalar_f8_co, *args: _Scalar_f8_co, loc: _Scalar_f8_co = 0, **kwds: _Scalar_f8_co) -> _Scalar_f8: ...
    @overload
    def logcdf(
        self,
        /,
        k: _Arr_f8_co[_ShapeT],
        *args: _Scalar_f8_co,
        loc: _Scalar_f8_co = 0,
        **kwds: _Scalar_f8_co,
    ) -> _Arr_f8[_ShapeT]: ...
    @overload
    def logcdf(
        self,
        /,
        k: _ArrLike_f8_co,
        *args: _ArrLike_f8_co,
        loc: _ArrLike_f8_co = 0,
        **kwds: _ArrLike_f8_co,
    ) -> _ArrLike_f8: ...

    #
    @overload
    def sf(self, /, k: _Scalar_f8_co, *args: _Scalar_f8_co, loc: _Scalar_f8_co = 0, **kwds: _Scalar_f8_co) -> _Scalar_f8: ...
    @overload
    def sf(
        self,
        /,
        k: _Arr_f8_co[_ShapeT],
        *args: _Scalar_f8_co,
        loc: _Scalar_f8_co = 0,
        **kwds: _Scalar_f8_co,
    ) -> _Arr_f8[_ShapeT]: ...
    @overload
    def sf(
        self,
        /,
        k: _ArrLike_f8_co,
        *args: _ArrLike_f8_co,
        loc: _ArrLike_f8_co = 0,
        **kwds: _ArrLike_f8_co,
    ) -> _ArrLike_f8: ...
    #
    @overload
    def logsf(self, /, k: _Scalar_f8_co, *args: _Scalar_f8_co, loc: _Scalar_f8_co = 0, **kwds: _Scalar_f8_co) -> _Scalar_f8: ...
    @overload
    def logsf(
        self,
        /,
        k: _Arr_f8_co[_ShapeT],
        *args: _Scalar_f8_co,
        loc: _Scalar_f8_co = 0,
        **kwds: _Scalar_f8_co,
    ) -> _Arr_f8[_ShapeT]: ...
    @overload
    def logsf(
        self,
        /,
        k: _ArrLike_f8_co,
        *args: _ArrLike_f8_co,
        loc: _ArrLike_f8_co = 0,
        **kwds: _ArrLike_f8_co,
    ) -> _ArrLike_f8: ...

    #
    @overload
    def ppf(self, /, q: _Scalar_f8_co, *args: _Scalar_f8_co, loc: _Scalar_f8_co = 0, **kwds: _Scalar_f8_co) -> _Scalar_f8: ...
    @overload
    def ppf(
        self,
        /,
        q: _Arr_f8_co[_ShapeT],
        *args: _Scalar_f8_co,
        loc: _Scalar_f8_co = 0,
        **kwds: _Scalar_f8_co,
    ) -> _Arr_f8[_ShapeT]: ...
    @overload
    def ppf(
        self,
        /,
        q: _ArrLike_f8_co,
        *args: _ArrLike_f8_co,
        loc: _ArrLike_f8_co = 0,
        **kwds: _ArrLike_f8_co,
    ) -> _ArrLike_f8: ...
    #
    @overload
    def isf(self, /, q: _Scalar_f8_co, *args: _Scalar_f8_co, loc: _Scalar_f8_co = 0, **kwds: _Scalar_f8_co) -> _Scalar_f8: ...
    @overload
    def isf(
        self,
        /,
        q: _Arr_f8_co[_ShapeT],
        *args: _Scalar_f8_co,
        loc: _Scalar_f8_co = 0,
        **kwds: _Scalar_f8_co,
    ) -> _Arr_f8[_ShapeT]: ...
    @overload
    def isf(
        self,
        /,
        q: _ArrLike_f8_co,
        *args: _ArrLike_f8_co,
        loc: _ArrLike_f8_co = 0,
        **kwds: _ArrLike_f8_co,
    ) -> _ArrLike_f8: ...

    #
    def expect(
        self,
        /,
        func: Callable[[onp.ArrayND[np.int_]], _Arr_f8_co] | None = None,
        args: tuple[_Scalar_f8_co, ...] = (),
        loc: _Scalar_f8_co = 0,
        lb: onp.ToInt | None = None,
        ub: onp.ToInt | None = None,
        conditional: op.CanBool = False,
        maxcount: onp.ToInt = 1000,
        tolerance: _Scalar_f8_co = 1e-10,
        chunksize: onp.ToInt = 32,
    ) -> _Scalar_f8: ...

    #
    @override
    def rvs(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        /,
        *args: _ArrLike_f8_co,
        loc: _ArrLike_f8_co = 0,
        size: spt.AnyShape = 1,
        random_state: spt.Seed | None = None,
        **kwds: _ArrLike_f8_co,
    ) -> _ArrLike_i8: ...

_XKT_co = TypeVar("_XKT_co", bound=np.number[Any], covariant=True, default=np.number[Any])
_PKT_co = TypeVar("_PKT_co", bound=_Scalar_f, covariant=True, default=_Scalar_f)

class rv_sample(rv_discrete, Generic[_XKT_co, _PKT_co]):
    xk: onp.Array1D[_XKT_co]
    pk: onp.Array1D[_PKT_co]
    qvals: onp.Array1D[_PKT_co]
    def __init__(  # pyright: ignore[reportInconsistentConstructor]
        self,
        /,
        a: _Scalar_f8_co = 0,
        b: _Scalar_f8_co = ...,
        name: LiteralString | None = None,
        badvalue: float | None = None,
        moment_tol: float = 1e-08,
        values: tuple[_ArrLike_f8_co, _ArrLike_f8_co] | None = None,
        inc: int = 1,
        longname: LiteralString | None = None,
        shapes: LiteralString | None = None,
        seed: spt.Seed | None = None,
    ) -> None: ...
    def _entropy(self, /) -> _Scalar_f8: ...
    vecentropy: Final = _entropy
    @override
    def generic_moment(self, /, n: onp.ToInt | onp.ToIntND | int | Sequence[int]) -> _Arr_f8: ...  # pyright: ignore[reportIncompatibleMethodOverride]

def get_distribution_names(namespace_pairs: Iterable[tuple[str, type]], rv_base_class: type) -> _Tuple2[list[LiteralString]]: ...

# private helper subtypes
@type_check_only
class _rv_continuous_0(rv_continuous):
    # overrides of rv_generic
    @override
    @overload
    def stats(self, /, loc: _Scalar_f8_co, scale: _Scalar_f8_co, moment: _Moments1) -> _Scalar_f8: ...
    @overload
    def stats(self, /, loc: _ArrLike_f8_co, scale: _ArrLike_f8_co, moment: _Moments1) -> _ArrLike_f8: ...
    @overload
    def stats(self, /, loc: _Scalar_f8_co = 0, scale: _Scalar_f8_co = 1, *, moment: _Moments1) -> _Scalar_f8: ...
    @overload
    def stats(self, /, loc: _ArrLike_f8_co = 0, scale: _ArrLike_f8_co = 1, *, moment: _Moments1) -> _ArrLike_f8: ...
    @overload
    def stats(self, /, loc: _Scalar_f8_co = 0, scale: _Scalar_f8_co = 1, moment: _Moments2 = ...) -> _Tuple2[_Scalar_f8]: ...
    @overload
    def stats(self, /, loc: _ArrLike_f8_co = 0, scale: _ArrLike_f8_co = 1, moment: _Moments2 = ...) -> _Tuple2[_ArrLike_f8]: ...
    @overload
    def stats(self, /, loc: _Scalar_f8_co, scale: _Scalar_f8_co, moment: _Moments3) -> _Tuple3[_Scalar_f8]: ...
    @overload
    def stats(self, /, loc: _ArrLike_f8_co, scale: _ArrLike_f8_co, moment: _Moments3) -> _Tuple3[_ArrLike_f8]: ...
    @overload
    def stats(self, /, loc: _Scalar_f8_co = 0, scale: _Scalar_f8_co = 1, *, moment: _Moments3) -> _Tuple3[_Scalar_f8]: ...
    @overload
    def stats(self, /, loc: _ArrLike_f8_co = 0, scale: _ArrLike_f8_co = 1, *, moment: _Moments3) -> _Tuple3[_ArrLike_f8]: ...
    @overload
    def stats(self, /, loc: _Scalar_f8_co, scale: _Scalar_f8_co, moment: _Moments4) -> _Tuple4[_Scalar_f8]: ...
    @overload
    def stats(self, /, loc: _ArrLike_f8_co, scale: _ArrLike_f8_co, moment: _Moments4) -> _Tuple4[_ArrLike_f8]: ...
    @overload
    def stats(self, /, loc: _Scalar_f8_co = 0, scale: _Scalar_f8_co = 1, *, moment: _Moments4) -> _Tuple4[_Scalar_f8]: ...
    @overload
    def stats(self, /, loc: _ArrLike_f8_co = 0, scale: _ArrLike_f8_co = 1, *, moment: _Moments4) -> _Tuple4[_ArrLike_f8]: ...  # pyright: ignore[reportIncompatibleMethodOverride]

    #
    @override
    @overload
    def entropy(self, /, loc: _Scalar_f8_co = 0, scale: _Scalar_f8_co = 1) -> _Scalar_f8: ...
    @overload
    def entropy(self, /, loc: _ArrLike_f8_co = 0, scale: _ArrLike_f8_co = 1) -> _ArrLike_f8: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    #
    @override
    @overload
    def moment(self, /, order: int | _Scalar_i, loc: _Scalar_f8_co = 0, scale: _Scalar_f8_co = 1) -> _Scalar_f8: ...
    @overload
    def moment(self, /, order: int | _Scalar_i, loc: _ArrLike_f8_co = 0, scale: _ArrLike_f8_co = 1) -> _ArrLike_f8: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    #
    @override
    @overload
    def median(self, /, loc: _Scalar_f8_co = 0, scale: _Scalar_f8_co = 1) -> _Scalar_f8: ...
    @overload
    def median(self, /, loc: _ArrLike_f8_co = 0, scale: _ArrLike_f8_co = 1) -> _ArrLike_f8: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    #
    @override
    @overload
    def mean(self, /, loc: _Scalar_f8_co = 0, scale: _Scalar_f8_co = 1) -> _Scalar_f8: ...
    @overload
    def mean(self, /, loc: _ArrLike_f8_co = 0, scale: _ArrLike_f8_co = 1) -> _ArrLike_f8: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    #
    @override
    @overload
    def var(self, /, loc: _Scalar_f8_co = 0, scale: _Scalar_f8_co = 1) -> _Scalar_f8: ...
    @overload
    def var(self, /, loc: _ArrLike_f8_co = 0, scale: _ArrLike_f8_co = 1) -> _ArrLike_f8: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    #
    @override
    @overload
    def std(self, /, loc: _Scalar_f8_co = 0, scale: _Scalar_f8_co = 1) -> _Scalar_f8: ...
    @overload
    def std(self, /, loc: _ArrLike_f8_co = 0, scale: _ArrLike_f8_co = 1) -> _ArrLike_f8: ...  # pyright: ignore[reportIncompatibleMethodOverride]

    #
    @override
    @overload
    def interval(self, /, confidence: _Scalar_f8_co, loc: _Scalar_f8_co = 0, scale: _Scalar_f8_co = 1) -> _Tuple2[_Scalar_f8]: ...
    @overload
    def interval(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        /,
        confidence: _ArrLike_f8_co,
        loc: _ArrLike_f8_co = 0,
        scale: _ArrLike_f8_co = 1,
    ) -> _Tuple2[_Scalar_f8] | _Tuple2[_Arr_f8]: ...
    #
    @override
    @overload
    def support(self, /, loc: _Scalar_f8_co = 0, scale: _Scalar_f8_co = 1) -> _Tuple2[_Scalar_f8]: ...
    @overload
    def support(self, /, loc: _ArrLike_f8_co = 0, scale: _ArrLike_f8_co = 1) -> _Tuple2[_Scalar_f8] | _Tuple2[_Arr_f8]: ...  # pyright: ignore[reportIncompatibleMethodOverride]

    # overrides of rv_continuous
    @override
    @overload
    def __call__(self, /) -> rv_continuous_frozen[Self, _Scalar_f8]: ...
    @overload
    def __call__(self, /, loc: _Scalar_f8_co = 0, scale: _Scalar_f8_co = 1) -> rv_continuous_frozen[Self, _Scalar_f8]: ...
    @overload
    def __call__(self, /, loc: _ArrLike_f8_co = 0, scale: _ArrLike_f8_co = 1) -> rv_continuous_frozen[Self]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    #
    @override
    @overload
    def freeze(self, /) -> rv_continuous_frozen[Self, _Scalar_f8]: ...
    @overload
    def freeze(self, /, loc: _Scalar_f8_co = 0, scale: _Scalar_f8_co = 1) -> rv_continuous_frozen[Self, _Scalar_f8]: ...
    @overload
    def freeze(self, /, loc: _ArrLike_f8_co = 0, scale: _ArrLike_f8_co = 1) -> rv_continuous_frozen[Self]: ...  # pyright: ignore[reportIncompatibleMethodOverride]

    #
    @override
    @overload
    def pdf(self, /, x: _Scalar_f8_co, loc: _Scalar_f8_co = 0, scale: _Scalar_f8_co = 1) -> _Scalar_f8: ...
    @overload
    def pdf(self, /, x: _Arr_f8_co[_ShapeT], loc: _Scalar_f8_co = 0, scale: _Scalar_f8_co = 1) -> _Arr_f8[_ShapeT]: ...
    @overload
    def pdf(self, /, x: _ArrLike_f8_co, loc: _ArrLike_f8_co = 0, scale: _ArrLike_f8_co = 1) -> _ArrLike_f8: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    #
    @override
    @overload
    def logpdf(self, /, x: _Scalar_f8_co, loc: _Scalar_f8_co = 0, scale: _Scalar_f8_co = 1) -> _Scalar_f8: ...
    @overload
    def logpdf(self, /, x: _Arr_f8_co[_ShapeT], loc: _Scalar_f8_co = 0, scale: _Scalar_f8_co = 1) -> _Arr_f8[_ShapeT]: ...
    @overload
    def logpdf(self, /, x: _ArrLike_f8_co, loc: _ArrLike_f8_co = 0, scale: _ArrLike_f8_co = 1) -> _ArrLike_f8: ...  # pyright: ignore[reportIncompatibleMethodOverride]

    #
    @override
    @overload
    def cdf(self, /, x: _Scalar_f8_co, loc: _Scalar_f8_co = 0, scale: _Scalar_f8_co = 1) -> _Scalar_f8: ...
    @overload
    def cdf(self, /, x: _Arr_f8_co[_ShapeT], loc: _Scalar_f8_co = 0, scale: _Scalar_f8_co = 1) -> _Arr_f8[_ShapeT]: ...
    @overload
    def cdf(self, /, x: _ArrLike_f8_co, loc: _ArrLike_f8_co = 0, scale: _ArrLike_f8_co = 1) -> _ArrLike_f8: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    #
    @override
    @overload
    def logcdf(self, /, x: _Scalar_f8_co, loc: _Scalar_f8_co = 0, scale: _Scalar_f8_co = 1) -> _Scalar_f8: ...
    @overload
    def logcdf(self, /, x: _Arr_f8_co[_ShapeT], loc: _Scalar_f8_co = 0, scale: _Scalar_f8_co = 1) -> _Arr_f8[_ShapeT]: ...
    @overload
    def logcdf(self, /, x: _ArrLike_f8_co, loc: _ArrLike_f8_co = 0, scale: _ArrLike_f8_co = 1) -> _ArrLike_f8: ...  # pyright: ignore[reportIncompatibleMethodOverride]

    #
    @override
    @overload
    def sf(self, /, x: _Scalar_f8_co, loc: _Scalar_f8_co = 0, scale: _Scalar_f8_co = 1) -> _Scalar_f8: ...
    @overload
    def sf(self, /, x: _Arr_f8_co[_ShapeT], loc: _Scalar_f8_co = 0, scale: _Scalar_f8_co = 1) -> _Arr_f8[_ShapeT]: ...
    @overload
    def sf(self, /, x: _ArrLike_f8_co, loc: _ArrLike_f8_co = 0, scale: _ArrLike_f8_co = 1) -> _ArrLike_f8: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    #
    @override
    @overload
    def logsf(self, /, x: _Scalar_f8_co, loc: _Scalar_f8_co = 0, scale: _Scalar_f8_co = 1) -> _Scalar_f8: ...
    @overload
    def logsf(self, /, x: _Arr_f8_co[_ShapeT], loc: _Scalar_f8_co = 0, scale: _Scalar_f8_co = 1) -> _Arr_f8[_ShapeT]: ...
    @overload
    def logsf(self, /, x: _ArrLike_f8_co, loc: _ArrLike_f8_co = 0, scale: _ArrLike_f8_co = 1) -> _ArrLike_f8: ...  # pyright: ignore[reportIncompatibleMethodOverride]

    #
    @override
    @overload
    def ppf(self, /, q: _Scalar_f8_co, loc: _Scalar_f8_co = 0, scale: _Scalar_f8_co = 1) -> _Scalar_f8: ...
    @overload
    def ppf(self, /, q: _Arr_f8_co[_ShapeT], loc: _Scalar_f8_co = 0, scale: _Scalar_f8_co = 1) -> _Arr_f8[_ShapeT]: ...
    @overload
    def ppf(self, /, q: _ArrLike_f8_co, loc: _ArrLike_f8_co = 0, scale: _ArrLike_f8_co = 1) -> _ArrLike_f8: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    #
    @override
    @overload
    def isf(self, /, q: _Scalar_f8_co, loc: _Scalar_f8_co = 0, scale: _Scalar_f8_co = 1) -> _Scalar_f8: ...
    @overload
    def isf(self, /, q: _Arr_f8_co[_ShapeT], loc: _Scalar_f8_co = 0, scale: _Scalar_f8_co = 1) -> _Arr_f8[_ShapeT]: ...
    @overload
    def isf(self, /, q: _ArrLike_f8_co, loc: _ArrLike_f8_co = 0, scale: _ArrLike_f8_co = 1) -> _ArrLike_f8: ...  # pyright: ignore[reportIncompatibleMethodOverride]

    #
    @override
    def rvs(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        /,
        loc: _Scalar_f8_co = 0,
        scale: _Scalar_f8_co = 1,
        size: spt.AnyShape = 1,
        random_state: spt.Seed | None = None,
    ) -> _ArrLike_f8: ...
