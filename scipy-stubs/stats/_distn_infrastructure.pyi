import abc
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Final, Generic, Literal, TypeAlias, overload, type_check_only
from typing_extensions import LiteralString, Self, TypeVar, Unpack, override

import numpy as np
import numpy.typing as npt
import optype.numpy as onpt
import scipy._typing as spt

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
_PT_co = TypeVar("_PT_co", bound=np.floating[Any], covariant=True, default=np.floating[Any])

_ShapeT = TypeVar("_ShapeT", bound=tuple[int, ...])

_Scalar_uif: TypeAlias = np.integer[Any] | np.floating[Any]
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

_StatsMoment: TypeAlias = Literal["m", "v", "s", "k", "mv", "ms", "mk", "vs", "vk", "sk", "mvs", "mvk", "msk", "vsk", "mvsk"]
_FitMethod: TypeAlias = Literal["MLE", "MM"]

###

parse_arg_template: Final[str]

def argsreduce(cond: npt.NDArray[np.bool_], *args: npt.ArrayLike) -> list[npt.NDArray[np.floating[Any] | np.integer[Any]]]: ...

class rv_frozen(Generic[_RVG_co]):
    args: Final[_RVArgs]
    kwds: Final[_RVKwds]
    @property
    def dist(self, /) -> _RVG_co: ...
    @property
    def random_state(self, /) -> _RNG: ...
    @random_state.setter
    def random_state(self, seed: _Seed, /) -> None: ...
    def __init__(self, dist: _RVG_co, /, *args: _AnyArray_f8_in, **kwds: _AnyArray_f8_in) -> None: ...
    def cdf(self, /, x: _AnyArray_f8_in) -> _AnyArray_f8_out: ...
    def logcdf(self, /, x: _AnyArray_f8_in) -> _AnyArray_f8_out: ...
    def ppf(self, /, q: _AnyArray_f8_in) -> _AnyArray_f8_out: ...
    def isf(self, /, q: _AnyArray_f8_in) -> _AnyArray_f8_out: ...
    def rvs(
        self,
        /,
        size: spt.AnyInt | tuple[spt.AnyInt, ...] | None = None,
        random_state: _Seed | None = None,
    ) -> _AnyArray_f8_out: ...
    def sf(self, /, x: _AnyArray_f8_in) -> _AnyArray_f8_out: ...
    def logsf(self, /, x: _AnyArray_f8_in) -> _AnyArray_f8_out: ...
    @overload
    def stats(
        self,
        /,
        moments: Literal["m", "v", "s", "k"],
    ) -> tuple[np.float64] | tuple[npt.NDArray[np.float64]]: ...
    @overload
    def stats(
        self,
        /,
        moments: Literal["mv", "ms", "mk", "vs", "vk", "sk"] = ...,
    ) -> tuple[np.float64, np.float64] | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
    @overload
    def stats(
        self,
        /,
        moments: Literal["mvs", "mvk", "msk", "vsk"],
    ) -> (
        tuple[np.float64, np.float64, np.float64]
        | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]
    ): ...
    @overload
    def stats(
        self,
        /,
        moments: Literal["mvsk"],
    ) -> (
        tuple[np.float64, np.float64, np.float64, np.float64]
        | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]
    ): ...
    def median(self, /) -> _AnyArray_f8_out: ...
    def mean(self, /) -> _AnyArray_f8_out: ...
    def var(self, /) -> _AnyArray_f8_out: ...
    def std(self, /) -> _AnyArray_f8_out: ...
    # order defaults to `None`, but that will `raise TypeError`
    def moment(self, /, order: int) -> np.float64: ...
    def entropy(self, /) -> _AnyArray_f8_out: ...
    def interval(
        self,
        /,
        confidence: _AnyScalar_f8_in | None = None,
    ) -> tuple[np.float64, np.float64] | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
    def support(self, /) -> tuple[np.float64, np.float64] | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
    # requires all args to be scalars
    def expect(
        self,
        /,
        func: Callable[[float], float] | None = None,
        lb: _AnyScalar_f8_in | None = None,
        ub: _AnyScalar_f8_in | None = None,
        conditional: bool = False,
        # TODO: use `TypedDict` and `Unpack` with the `scipy.integrate.quad` kwargs
        **kwds: Any,
    ) -> np.float64: ...

class rv_continuous_frozen(rv_frozen[_RVC_co], Generic[_RVC_co]):
    def pdf(self, /, x: _AnyArray_f8_in) -> _AnyArray_f8_out: ...
    def logpdf(self, /, x: _AnyArray_f8_in) -> _AnyArray_f8_out: ...

class rv_discrete_frozen(rv_frozen[_RVD_co], Generic[_RVD_co]):
    def pmf(self, /, k: _AnyArray_f8_in) -> _AnyArray_f8_out: ...
    def logpmf(self, /, k: _AnyArray_f8_in) -> _AnyArray_f8_out: ...

_ParamT = TypeVar("_ParamT", bound=_Scalar_uif)
# _LocT = TypeVar("_LocT", bound=_Scalar_uif)
# _ScaleT = TypeVar("_ScaleT", bound=_Scalar_uif)

class rv_generic:
    # TODO: private methods
    def __init__(self, /, seed: _Seed | None = None) -> None: ...
    @property
    def random_state(self, /) -> _RNG: ...
    @random_state.setter
    def random_state(self, seed: _Seed, /) -> None: ...
    @abc.abstractmethod
    def _unpack_loc_scale(
        self,
        /,
        theta: Sequence[_AnyArray_f8_in],
    ) -> tuple[_AnyArray_f8_in, _AnyArray_f8_in, tuple[_AnyArray_f8_in]]: ...
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
    def freeze(self, /, *args: _AnyArray_f8_in, **kwds: _AnyArray_f8_in) -> rv_frozen[Self]: ...
    def __call__(self, /, *args: _AnyArray_f8_in, **kwds: _AnyArray_f8_in) -> rv_frozen[Self]: ...
    def _stats(
        self,
        /,
        *args: _AnyArray_f8_in,
        **kwargs: _AnyArray_f8_in,
    ) -> tuple[_AnyArray_f8_out | None, _AnyArray_f8_out | None, _AnyArray_f8_out | None, _AnyArray_f8_out | None]: ...
    def _munp(self, /, n: onpt.AnyIntegerArray, *args: _AnyScalar_f8_in) -> npt.NDArray[np.float64]: ...
    # TODO: see: https://github.com/KotlinIsland/basedmypy/issues/747
    # ruff: noqa: ERA001
    # @overload
    # def _argcheck_rvs(
    #     self,
    #     /,
    #     # NOTE: This `Unpack` trick is only understood by pyright; mypy (still) doesn't fully support `Unpack` (`mypy<=1.11.1`)
    #     *args: Unpack[tuple[Unpack[tuple[_ParamT, ...]], _LocT, _ScaleT]],
    #     size: onpt.AnyIntegerArray | None = None,
    # ) -> tuple[
    #     list[npt.NDArray[_ParamT]],
    #     npt.NDArray[_LocT],
    #     npt.NDArray[_ScaleT],
    #     tuple[int, ...] | tuple[np.intp, ...],
    # ]: ...
    # @overload
    # def _argcheck_rvs(
    #     self,
    #     /,
    #     *args: Unpack[tuple[Unpack[tuple[int, ...]], int, int]],
    #     size: onpt.AnyIntegerArray | None = None,
    # ) -> tuple[
    #     list[npt.NDArray[np.intp]],
    #     npt.NDArray[np.intp],
    #     npt.NDArray[np.intp],
    #     tuple[int, ...] | tuple[np.intp, ...],
    # ]: ...
    # @overload
    # def _argcheck_rvs(
    #     self,
    #     /,
    #     *args: Unpack[tuple[Unpack[tuple[float, ...]], float, float]],
    #     size: onpt.AnyIntegerArray | None = None,
    # ) -> tuple[
    #     # NOTE: this first union type shouldn't be needed, but is required to work around a pyright bug
    #     list[npt.NDArray[np.intp]] | list[npt.NDArray[np.intp | np.float64]],
    #     npt.NDArray[np.intp | np.float64],
    #     npt.NDArray[np.intp | np.float64],
    #     tuple[int, ...] | tuple[np.intp, ...],
    # ]: ...
    @overload
    def _argcheck_rvs(
        self,
        /,
        *args: tuple[_ParamT, ...],
        size: onpt.AnyIntegerArray | None = None,
    ) -> tuple[
        list[npt.NDArray[_ParamT]],
        npt.NDArray[_ParamT],
        npt.NDArray[_ParamT],
        tuple[int, ...] | tuple[np.intp, ...],
    ]: ...
    @overload
    def _argcheck_rvs(
        self,
        /,
        *args: tuple[int, ...],
        size: onpt.AnyIntegerArray | None = None,
    ) -> tuple[
        list[npt.NDArray[np.intp]],
        npt.NDArray[np.intp],
        npt.NDArray[np.intp],
        tuple[int, ...] | tuple[np.intp, ...],
    ]: ...
    @overload
    def _argcheck_rvs(
        self,
        /,
        *args: tuple[float, ...],
        size: onpt.AnyIntegerArray | None = None,
    ) -> tuple[
        # NOTE: this first union type shouldn't be needed, but is required to work around a pyright bug
        list[npt.NDArray[np.intp]] | list[npt.NDArray[np.intp | np.float64]],
        npt.NDArray[np.intp | np.float64],
        npt.NDArray[np.intp | np.float64],
        tuple[int, ...] | tuple[np.intp, ...],
    ]: ...
    def _argcheck(self, /, *args: _AnyArray_f8_in) -> npt.NDArray[np.bool_]: ...
    def _get_support(self, /, *args: _AnyArray_f8_in, **kwargs: _AnyArray_f8_in) -> tuple[_AnyArray_f8_in, _AnyArray_f8_in]: ...
    def _support_mask(
        self,
        x: npt.NDArray[_Scalar_uif],
        /,
        *args: float | _Scalar_uif | npt.NDArray[_Scalar_uif],
    ) -> npt.NDArray[np.bool_]: ...
    def _open_support_mask(
        self,
        x: npt.NDArray[_Scalar_uif],
        /,
        *args: float | _Scalar_uif | npt.NDArray[_Scalar_uif],
    ) -> npt.NDArray[np.bool_]: ...
    def _rvs(
        self,
        /,
        *args: _AnyArray_f8_in,
        size: int | tuple[int, ...] | None = None,
        random_state: _Seed | None = None,
    ) -> _AnyArray_f8_out: ...
    def _logcdf(self, x: npt.NDArray[_Scalar_uif], /, *args: npt.NDArray[_Scalar_uif]) -> npt.NDArray[np.float64]: ...
    def _sf(self, x: npt.NDArray[_Scalar_uif], /, *args: npt.NDArray[_Scalar_uif]) -> npt.NDArray[np.float64]: ...
    def _logsf(self, x: npt.NDArray[_Scalar_uif], /, *args: npt.NDArray[_Scalar_uif]) -> npt.NDArray[np.float64]: ...
    def _ppf(self, q: npt.NDArray[_Scalar_uif], /, *args: npt.NDArray[_Scalar_uif]) -> npt.NDArray[np.float64]: ...
    def _isf(self, q: npt.NDArray[_Scalar_uif], /, *args: npt.NDArray[_Scalar_uif]) -> npt.NDArray[np.float64]: ...
    @overload
    def rvs(
        self,
        /,
        *args: npt.ArrayLike,
        random_state: _Seed,
        discrete: Literal[True],
        **kwds: _AnyArray_f8_in,
    ) -> int | npt.NDArray[np.int64]: ...  # NOTE: this is `int64`; not `intp`
    @overload
    def rvs(
        self,
        /,
        *args: npt.ArrayLike,
        random_state: _Seed,
        discrete: Literal[False, None] = ...,
        **kwds: _AnyArray_f8_in,
    ) -> _AnyArray_f8_out: ...
    @overload
    def stats(
        self,
        /,
        *args: _AnyScalar_f8_in,
        moment: _StatsMoment = ...,
        **kwds: _AnyScalar_f8_in,
    ) -> tuple[np.float64, ...]: ...
    @overload
    def stats(
        self,
        /,
        *args: _AnyArray_f8_in,
        moment: _StatsMoment = ...,
        **kwds: _AnyArray_f8_in,
    ) -> tuple[np.float64, ...] | tuple[npt.NDArray[np.float64], ...]: ...
    @overload
    def entropy(self, /) -> np.float64: ...
    @overload
    def entropy(self, /, *args: _AnyScalar_f8_in, **kwds: _AnyScalar_f8_in) -> np.float64: ...
    @overload
    def entropy(self, /, *args: _AnyArray_f8_in, **kwds: _AnyArray_f8_in) -> _AnyArray_f8_out: ...
    @overload
    def moment(self, order: int, /, *args: _AnyScalar_f8_in, **kwds: _AnyScalar_f8_in) -> np.float64: ...
    @overload
    def moment(self, order: int, /, *args: _AnyArray_f8_in, **kwds: _AnyArray_f8_in) -> _AnyArray_f8_out: ...
    @overload
    def median(self, /, *args: _AnyScalar_f8_in, **kwds: _AnyScalar_f8_in) -> np.float64: ...
    @overload
    def median(self, /, *args: _AnyArray_f8_in, **kwds: _AnyArray_f8_in) -> _AnyArray_f8_out: ...
    @overload
    def mean(self, /, *args: _AnyScalar_f8_in, **kwds: _AnyScalar_f8_in) -> np.float64: ...
    @overload
    def mean(self, /, *args: _AnyArray_f8_in, **kwds: _AnyArray_f8_in) -> _AnyArray_f8_out: ...
    @overload
    def var(self, /, *args: _AnyScalar_f8_in, **kwds: _AnyScalar_f8_in) -> np.float64: ...
    @overload
    def var(self, /, *args: _AnyArray_f8_in, **kwds: _AnyArray_f8_in) -> _AnyArray_f8_out: ...
    @overload
    def std(self, /, *args: _AnyScalar_f8_in, **kwds: _AnyScalar_f8_in) -> np.float64: ...
    @overload
    def std(self, /, *args: _AnyArray_f8_in, **kwds: _AnyArray_f8_in) -> _AnyArray_f8_out: ...
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
    def support(self, /, *args: _AnyScalar_f8_in, **kwds: _AnyScalar_f8_in) -> tuple[np.float64, np.float64]: ...
    @overload
    def support(
        self,
        /,
        *args: _AnyArray_f8_in,
        **kwds: _AnyArray_f8_in,
    ) -> tuple[np.float64, np.float64] | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
    def nnlf(self, /, theta: Sequence[_Scalar_f8_in], x: onpt.AnyIntegerArray | onpt.AnyFloatingArray) -> _AnyArray_f8_out: ...
    def _nnlf(self, x: npt.NDArray[np.floating[Any]], /, *args: _Scalar_f8_in) -> _AnyArray_f8_out: ...
    def _nnlf_and_penalty(
        self,
        /,
        x: npt.NDArray[_Scalar_uif],
        args: Sequence[_Scalar_f8_in],
        log_fitfun: Callable[..., npt.NDArray[np.float64]],
    ) -> np.float64: ...
    def _penalized_nnlf(
        self,
        /,
        theta: Sequence[_Scalar_f8_in],
        x: npt.NDArray[_Scalar_uif],
    ) -> np.float64: ...
    def _penalized_nlpsf(
        self,
        /,
        theta: Sequence[_Scalar_f8_in],
        x: npt.NDArray[_Scalar_uif],
    ) -> np.float64: ...

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
    def _attach_methods(self, /) -> None: ...
    def generic_moment(self, /, n: onpt.AnyIntegerArray, *args: _Scalar_f8_in) -> npt.NDArray[np.float64]: ...
    def _logpxf(self, x: npt.NDArray[_Scalar_uif], /, *args: npt.NDArray[_Scalar_uif]) -> npt.NDArray[np.float64]: ...
    def _cdf_single(self, x: _Scalar_f8_in, /, *args: _Scalar_f8_in) -> np.float64: ...
    def _cdfvec(self, x: npt.NDArray[_Scalar_uif], /, *args: npt.NDArray[_Scalar_uif]) -> npt.NDArray[np.float64]: ...
    def _cdf(self, x: npt.NDArray[_Scalar_uif], /, *args: npt.NDArray[_Scalar_uif]) -> npt.NDArray[np.float64]: ...
    def _ppfvec(self, q: npt.NDArray[_Scalar_uif], /, *args: npt.NDArray[_Scalar_uif]) -> npt.NDArray[np.float64]: ...
    @overload
    def ppf(self, q: _Scalar_f8_in, /, *args: _Scalar_f8_in) -> np.float64: ...
    @overload
    def ppf(self, q: _AnyArray_f8_in, /, *args: _AnyArray_f8_in, **kwds: _AnyArray_f8_in) -> _AnyArray_f8_out: ...
    @overload
    def isf(self, q: _Scalar_f8_in, /, *args: _Scalar_f8_in, **kwds: _Scalar_f8_in) -> np.float64: ...
    @overload
    def isf(self, q: _AnyArray_f8_in, /, *args: _AnyArray_f8_in, **kwds: _AnyArray_f8_in) -> _AnyArray_f8_out: ...
    @overload
    def cdf(self, x: _Scalar_f8_in, /, *args: _Scalar_f8_in, **kwds: _Scalar_f8_in) -> np.float64: ...
    @overload
    def cdf(self, x: _AnyArray_f8_in, /, *args: _AnyArray_f8_in, **kwds: _AnyArray_f8_in) -> _AnyArray_f8_out: ...
    @overload
    def logcdf(self, x: _Scalar_f8_in, /, *args: _Scalar_f8_in, **kwds: _Scalar_f8_in) -> np.float64: ...
    @overload
    def logcdf(self, x: _AnyArray_f8_in, /, *args: _AnyArray_f8_in, **kwds: _AnyArray_f8_in) -> _AnyArray_f8_out: ...
    @overload
    def sf(self, x: _Scalar_f8_in, /, *args: _Scalar_f8_in, **kwds: _Scalar_f8_in) -> np.float64: ...
    @overload
    def sf(self, x: _AnyArray_f8_in, /, *args: _AnyArray_f8_in, **kwds: _AnyArray_f8_in) -> _AnyArray_f8_out: ...
    @overload
    def logsf(self, x: _Scalar_f8_in, /, *args: _Scalar_f8_in, **kwds: _Scalar_f8_in) -> np.float64: ...
    @overload
    def logsf(self, x: _AnyArray_f8_in, /, *args: _AnyArray_f8_in, **kwds: _AnyArray_f8_in) -> _AnyArray_f8_out: ...
    def _unpack_loc_scale(
        self,
        /,
        theta: Sequence[_AnyArray_f8_in],
    ) -> tuple[_AnyArray_f8_in, _AnyArray_f8_in, tuple[_AnyArray_f8_in]]: ...

class rv_continuous(_rv_mixin, rv_generic):
    moment_type: Final[Literal[0, 1]]
    name: Final[LiteralString]
    a: Final[float]
    b: Final[float]
    badvalue: Final[float]
    xtol: Final[float]
    shapes: Final[LiteralString]

    def __init__(
        self,
        /,
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
    @override
    def __call__(self, /, *args: _AnyArray_f8_in, **kwds: _AnyArray_f8_in) -> rv_continuous_frozen[Self]: ...
    @override
    def freeze(self, /, *args: _AnyArray_f8_in, **kwds: _AnyArray_f8_in) -> rv_continuous_frozen[Self]: ...
    def _pdf(self, x: npt.NDArray[_Scalar_uif], /, *args: npt.NDArray[_Scalar_uif]) -> npt.NDArray[np.float64]: ...
    def _logpdf(self, x: npt.NDArray[_Scalar_uif], /, *args: npt.NDArray[_Scalar_uif]) -> npt.NDArray[np.float64]: ...
    @overload
    def pdf(self, x: _Scalar_f8_in, /, *args: _Scalar_f8_in, **kwds: _Scalar_f8_in) -> np.float64: ...
    @overload
    def pdf(self, x: _AnyArray_f8_in, /, *args: _AnyArray_f8_in, **kwds: _AnyArray_f8_in) -> _AnyArray_f8_out: ...
    @overload
    def logpdf(self, x: _Scalar_f8_in, /, *args: _Scalar_f8_in, **kwds: _Scalar_f8_in) -> np.float64: ...
    @overload
    def logpdf(self, x: _AnyArray_f8_in, /, *args: _AnyArray_f8_in, **kwds: _AnyArray_f8_in) -> _AnyArray_f8_out: ...
    def _fitstart(
        self,
        /,
        data: _AnyArray_f8_in,
        args: tuple[_AnyScalar_f8_in] | None = None,
    ) -> tuple[Unpack[tuple[_AnyScalar_f8_in, ...]], float, float]: ...
    def _reduce_func(
        self,
        /,
        args: tuple[float | _Scalar_uif, ...],
        kwds: dict[str, Any],
        data: _AnyArray_f8_in | None = None,
    ) -> tuple[
        list[float | _Scalar_uif],
        Callable[[list[float | _Scalar_uif], npt.NDArray[_Scalar_uif]], float | np.float64],
        Callable[[list[float | _Scalar_uif], npt.NDArray[_Scalar_uif]], list[float | _Scalar_uif]],
        list[float | _Scalar_uif],
    ]: ...
    def _moment_error(
        self,
        /,
        theta: Sequence[_AnyArray_f8_in],
        x: npt.NDArray[_Scalar_uif],
        data_moments: npt.NDArray[np.floating[Any]],
    ) -> np.float64: ...
    def fit(
        self,
        data: _AnyArray_f8_in,
        /,
        *args: _Scalar_f8_in,
        optimizer: Callable[
            [npt.NDArray[np.float64], tuple[np.float64, ...], tuple[np.float64, ...], bool],
            tuple[np.float64, ...],
        ],
        method: _FitMethod = "MLE",
        **kwds: _Scalar_f8_in,
    ) -> tuple[np.float64, ...]: ...
    def _fit_loc_scale_support(
        self,
        data: _AnyArray_f8_in,
        /,
        *args: _Scalar_f8_in,
    ) -> tuple[np.intp | np.float64, np.intp | np.float64 | float]: ...
    def fit_loc_scale(self, data: _AnyArray_f8_in, /, *args: _Scalar_f8_in) -> tuple[np.float64, np.float64]: ...
    def expect(
        self,
        /,
        func: Callable[[float], float] | None = None,
        args: tuple[_AnyScalar_f8_in, ...] = (),
        loc: _AnyScalar_f8_in = 0,
        scale: _AnyScalar_f8_in = 1,
        lb: spt.AnyReal | None = None,
        ub: spt.AnyReal | None = None,
        conditional: bool = False,
        # TODO: use `TypedDict` and `Unpack` with the `scipy.integrate.quad` kwargs
        **kwds: Any,
    ) -> np.float64: ...

class rv_discrete(_rv_mixin, rv_generic):
    inc: int
    moment_tol: float

    @overload
    def __new__(
        cls,
        /,
        *,
        name: LiteralString,
        a: float = 0,
        b: float = ...,
        badvalue: float | None = None,
        moment_tol: float = 1e-08,
        inc: int = 1,
        longname: LiteralString | None = None,
        shapes: LiteralString | None = None,
        seed: _Seed | None = None,
        values: None = None,
    ) -> Self: ...
    @overload
    def __new__(
        cls,
        /,
        *,
        name: LiteralString,
        values: tuple[_AnyArray_f8_in, _AnyArray_f8_in],
        a: float = 0,
        b: float = ...,
        badvalue: float | None = None,
        moment_tol: float = 1e-08,
        inc: int = 1,
        longname: LiteralString | None = None,
        shapes: LiteralString | None = None,
        seed: _Seed | None = None,
    ) -> rv_sample: ...
    def __init__(
        self,
        /,
        *,
        a: float = 0,
        b: float = ...,
        name: LiteralString | None = None,
        badvalue: float | None = None,
        moment_tol: float = 1e-08,
        inc: int = 1,
        longname: LiteralString | None = None,
        shapes: LiteralString | None = None,
        seed: _Seed | None = None,
    ) -> None: ...
    @override
    def __call__(self, /, *args: _AnyArray_f8_in, **kwds: _AnyArray_f8_in) -> rv_discrete_frozen[Self]: ...
    @override
    def freeze(self, /, *args: _AnyArray_f8_in, **kwds: _AnyArray_f8_in) -> rv_discrete_frozen[Self]: ...
    @override
    def rvs(  # type: ignore[override]
        self,
        /,
        *args: npt.ArrayLike,
        random_state: _Seed,
        **kwds: _AnyArray_f8_in,
    ) -> int | npt.NDArray[np.int64]: ...
    @overload
    def pmf(self, k: _AnyScalar_f8_in, /, *args: _AnyScalar_f8_in, **kwds: _AnyScalar_f8_in) -> np.float64: ...
    @overload
    def pmf(self, k: _AnyArray_f8_in, /, *args: _AnyArray_f8_in, **kwds: _AnyArray_f8_in) -> _AnyArray_f8_out: ...
    @overload
    def logpmf(self, k: _AnyScalar_f8_in, /, *args: _AnyScalar_f8_in, **kwds: _AnyScalar_f8_in) -> np.float64: ...
    @overload
    def logpmf(self, k: _AnyArray_f8_in, /, *args: _AnyArray_f8_in, **kwds: _AnyArray_f8_in) -> _AnyArray_f8_out: ...
    def expect(
        self,
        /,
        func: Callable[[np.ndarray[_ShapeT, np.dtype[np.intp]]], np.ndarray[_ShapeT, np.dtype[_Scalar_f8_in]]] | None = None,
        args: tuple[_AnyScalar_f8_in, ...] = (),
        loc: _AnyScalar_f8_in = 0,
        lb: spt.AnyInt | None = None,
        ub: spt.AnyInt | None = None,
        conditional: spt.AnyBool = False,
        maxcount: spt.AnyInt = 1000,
        tolerance: spt.AnyReal = 1e-10,
        chunksize: spt.AnyInt = 32,
    ) -> np.float64: ...

class rv_sample(rv_discrete, Generic[_XT_co, _PT_co]):
    badvalue: Final[float]
    shapes: Final = " "
    @property
    def xk(self, /) -> np.ndarray[tuple[int], np.dtype[_XT_co]]: ...
    @property
    def pk(self, /) -> np.ndarray[tuple[int], np.dtype[_PT_co]]: ...
    @property
    def qvals(self, /) -> np.ndarray[tuple[int], np.dtype[_PT_co]]: ...
    @property
    def a(self, /) -> _XT_co: ...
    @property
    def b(self, /) -> _XT_co: ...
    def __init__(
        self,
        /,
        *,
        values: tuple[_AnyArray_f8_in, _AnyArray_f8_in],
        badvalue: float | None = None,
        moment_tol: float = 1e-08,
        inc: int = 1,
        name: LiteralString | None = None,
        longname: LiteralString | None = None,
        seed: _Seed | None = None,
    ) -> None: ...
    def vecentropy(self, /) -> np.float64: ...

def get_distribution_names(
    namespace_pairs: Iterable[tuple[str, type]],
    rv_base_class: type,
) -> tuple[list[LiteralString], list[LiteralString]]: ...
