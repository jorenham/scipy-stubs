import abc
from collections.abc import Sequence
from typing import Any, Final, Generic, Literal, TypeAlias, overload, type_check_only
from typing_extensions import Self, TypeVar, override

import numpy as np
import numpy.typing as npt
import optype as op
import optype.numpy as onpt
import scipy._typing as spt
from scipy.stats import _covariance
from ._covariance import Covariance

__all__ = [
    "dirichlet",
    "dirichlet_multinomial",
    "invwishart",
    "matrix_normal",
    "multinomial",
    "multivariate_hypergeom",
    "multivariate_normal",
    "multivariate_t",
    "ortho_group",
    "random_correlation",
    "random_table",
    "special_ortho_group",
    "uniform_direction",
    "unitary_group",
    "vonmises_fisher",
    "wishart",
]

_RVG_co = TypeVar("_RVG_co", bound=multi_rv_generic, covariant=True, default=multi_rv_generic)
_RVF_co = TypeVar("_RVF_co", bound=multi_rv_frozen, covariant=True)

_Scalar_uif: TypeAlias = np.integer[Any] | np.floating[Any]
_ArrayLike_uif_1d: TypeAlias = (
    Sequence[float | _Scalar_uif]
    | onpt.CanArray[tuple[int], np.dtype[_Scalar_uif]]
    | npt.NDArray[_Scalar_uif]
)  # fmt: skip
_ArrayLike_uif_2d: TypeAlias = (
    Sequence[_ArrayLike_uif_1d]
    | onpt.CanArray[tuple[int, int], np.dtype[_Scalar_uif]]
    | npt.NDArray[_Scalar_uif]
)  # fmt: skip
_ArrayLike_uif_2d_max: TypeAlias = spt.AnyReal | _ArrayLike_uif_1d | _ArrayLike_uif_2d
_ArrayLike_uif_nd: TypeAlias = Sequence[float | _ArrayLike_uif_nd] | onpt.CanArray[Any, np.dtype[_Scalar_uif]]
_ArrayLike_ui_nd: TypeAlias = Sequence[int | _ArrayLike_ui_nd] | onpt.CanArray[Any, np.dtype[np.integer[Any]]]
_ArrayLike_f_nd: TypeAlias = Sequence[float | _ArrayLike_f_nd] | onpt.CanArray[Any, np.dtype[np.floating[Any]]]

_ScalarOrArray_f8: TypeAlias = np.float64 | onpt.Array[onpt.AtLeast1D, np.float64]

_AnyCov: TypeAlias = Covariance | _ArrayLike_uif_2d | spt.AnyReal

@type_check_only
class rng_mixin:
    @property
    def random_state(self, /) -> spt.RNG: ...
    @random_state.setter
    def random_state(self, /, seed: spt.Seed) -> None: ...

class multi_rv_generic(rng_mixin):
    def __init__(self, /, seed: spt.Seed | None = None) -> None: ...
    def _get_random_state(self, /, random_state: spt.Seed) -> spt.RNG: ...
    @abc.abstractmethod
    def __call__(self, /, *args: Any, **kwds: Any) -> multi_rv_frozen[Self]: ...

class multi_rv_frozen(rng_mixin, Generic[_RVG_co]):
    @property
    def _dist(self, /) -> _RVG_co: ...

class multivariate_normal_gen(multi_rv_generic):
    @override
    def __call__(
        self,
        /,
        mean: _ArrayLike_uif_1d | None = None,
        cov: _AnyCov = 1,
        allow_singular: bool = False,
        seed: spt.Seed | None = None,
    ) -> multivariate_normal_frozen: ...
    def logpdf(
        self,
        /,
        x: _ArrayLike_uif_nd,
        mean: _ArrayLike_uif_1d | None = None,
        cov: _AnyCov = 1,
        allow_singular: bool = False,
    ) -> _ScalarOrArray_f8: ...
    def pdf(
        self,
        /,
        x: _ArrayLike_uif_nd,
        mean: _ArrayLike_uif_1d | None = None,
        cov: _AnyCov = 1,
        allow_singular: bool = False,
    ) -> _ScalarOrArray_f8: ...
    def logcdf(
        self,
        /,
        x: _ArrayLike_uif_nd,
        mean: _ArrayLike_uif_1d | None = None,
        cov: _AnyCov = 1,
        allow_singular: bool = False,
        maxpts: int | None = None,
        abseps: float = 1e-05,
        releps: float = 1e-05,
        *,
        lower_limit: _ArrayLike_uif_1d | None = None,
    ) -> _ScalarOrArray_f8: ...
    def cdf(
        self,
        /,
        x: _ArrayLike_uif_nd,
        mean: _ArrayLike_uif_1d | None = None,
        cov: _AnyCov = 1,
        allow_singular: bool = False,
        maxpts: int | None = None,
        abseps: float = 1e-05,
        releps: float = 1e-05,
        *,
        lower_limit: _ArrayLike_uif_1d | None = None,
    ) -> _ScalarOrArray_f8: ...
    def rvs(
        self,
        /,
        mean: _ArrayLike_uif_1d | None = None,
        cov: _AnyCov = 1,
        size: int | tuple[int, ...] = 1,
        random_state: spt.Seed | None = None,
    ) -> npt.NDArray[np.float64]: ...
    def entropy(self, /, mean: _ArrayLike_uif_1d | None = None, cov: _AnyCov = 1) -> np.float64: ...
    def fit(
        self,
        x: _ArrayLike_uif_nd,
        fix_mean: _ArrayLike_uif_1d | None = None,
        fix_cov: _ArrayLike_uif_2d | None = None,
    ) -> tuple[onpt.Array[tuple[int], np.float64], onpt.Array[tuple[int, int], np.float64]]: ...

class multivariate_normal_frozen(multi_rv_frozen[multivariate_normal_gen]):
    dim: Final[int]
    allow_singular: Final[bool]
    maxpts: Final[int]
    abseps: Final[float]
    releps: Final[float]
    cov_object: Final[Covariance]
    def __init__(
        self,
        /,
        mean: _ArrayLike_uif_1d | None = None,
        cov: _AnyCov = 1,
        allow_singular: bool = False,
        seed: spt.Seed | None = None,
        maxpts: int | None = None,
        abseps: float = 1e-05,
        releps: float = 1e-05,
    ) -> None: ...
    @property
    def mean(self, /) -> onpt.Array[tuple[int], np.float64]: ...
    @property
    def cov(self, /) -> onpt.Array[tuple[int, int], np.float64]: ...
    def logpdf(self, /, x: _ArrayLike_uif_nd) -> _ScalarOrArray_f8: ...
    def pdf(self, /, x: _ArrayLike_uif_nd) -> _ScalarOrArray_f8: ...
    def logcdf(
        self,
        /,
        x: _ArrayLike_uif_nd,
        *,
        lower_limit: _ArrayLike_uif_1d | None = None,
    ) -> _ScalarOrArray_f8: ...
    def cdf(
        self,
        /,
        x: _ArrayLike_uif_nd,
        *,
        lower_limit: _ArrayLike_uif_1d | None = None,
    ) -> _ScalarOrArray_f8: ...
    def rvs(self, /, size: int | tuple[int, ...] = 1, random_state: spt.Seed | None = None) -> npt.NDArray[np.float64]: ...
    def entropy(self, /) -> np.float64: ...

class matrix_normal_gen(multi_rv_generic):
    @override
    def __call__(
        self,
        /,
        mean: _ArrayLike_uif_2d | None = None,
        rowcov: _ArrayLike_uif_2d | spt.AnyReal = 1,
        colcov: _ArrayLike_uif_2d | spt.AnyReal = 1,
        seed: spt.Seed | None = None,
    ) -> matrix_normal_frozen: ...
    def logpdf(
        self,
        /,
        X: _ArrayLike_uif_nd,
        mean: _ArrayLike_uif_2d | None = None,
        rowcov: _ArrayLike_uif_2d | spt.AnyReal = 1,
        colcov: _ArrayLike_uif_2d | spt.AnyReal = 1,
    ) -> _ScalarOrArray_f8: ...
    def pdf(
        self,
        /,
        X: _ArrayLike_uif_nd,
        mean: _ArrayLike_uif_2d | None = None,
        rowcov: _ArrayLike_uif_2d | spt.AnyReal = 1,
        colcov: _ArrayLike_uif_2d | spt.AnyReal = 1,
    ) -> _ScalarOrArray_f8: ...
    def rvs(
        self,
        /,
        mean: _ArrayLike_uif_2d | None = None,
        rowcov: _ArrayLike_uif_2d | spt.AnyReal = 1,
        colcov: _ArrayLike_uif_2d | spt.AnyReal = 1,
        size: op.typing.AnyInt = 1,
        random_state: spt.Seed | None = None,
    ) -> onpt.Array[tuple[int, int, int], np.float64]: ...
    def entropy(self, /, rowcov: _AnyCov = 1, colcov: _AnyCov = 1) -> np.float64: ...

class matrix_normal_frozen(multi_rv_frozen[matrix_normal_gen]):
    rowpsd: Final[_covariance._PSD]
    colpsd: Final[_covariance._PSD]
    def __init__(
        self,
        mean: _ArrayLike_uif_2d | None = None,
        rowcov: _ArrayLike_uif_2d | spt.AnyReal = 1,
        colcov: _ArrayLike_uif_2d | spt.AnyReal = 1,
        seed: spt.Seed | None = None,
    ) -> None: ...
    def logpdf(self, /, x: _ArrayLike_uif_nd) -> _ScalarOrArray_f8: ...
    def pdf(self, /, x: _ArrayLike_uif_nd) -> _ScalarOrArray_f8: ...
    def rvs(
        self,
        /,
        size: op.typing.AnyInt = 1,
        random_state: spt.Seed | None = None,
    ) -> onpt.Array[tuple[int, int, int], np.float64]: ...
    def entropy(self, /) -> np.float64: ...

class dirichlet_gen(multi_rv_generic):
    @override
    def __call__(self, /, alpha: _ArrayLike_uif_1d, seed: spt.Seed | None = None) -> dirichlet_frozen: ...
    def logpdf(self, /, x: _ArrayLike_uif_nd, alpha: _ArrayLike_uif_1d) -> _ScalarOrArray_f8: ...
    def pdf(self, /, x: _ArrayLike_uif_nd, alpha: _ArrayLike_uif_1d) -> _ScalarOrArray_f8: ...
    def mean(self, /, alpha: _ArrayLike_uif_1d) -> onpt.Array[tuple[int], np.float64]: ...
    def var(self, /, alpha: _ArrayLike_uif_1d) -> onpt.Array[tuple[int], np.float64]: ...
    def cov(self, /, alpha: _ArrayLike_uif_1d) -> onpt.Array[tuple[int, int], np.float64]: ...
    def entropy(self, /, alpha: _ArrayLike_uif_1d) -> np.float64: ...
    @overload
    def rvs(
        self,
        /,
        alpha: _ArrayLike_uif_1d,
        size: tuple[()],
        random_state: spt.Seed | None = None,
    ) -> onpt.Array[tuple[int], np.float64]: ...
    @overload
    def rvs(
        self,
        /,
        alpha: _ArrayLike_uif_1d,
        size: int | tuple[int] = 1,
        random_state: spt.Seed | None = None,
    ) -> onpt.Array[tuple[int, int], np.float64]: ...
    @overload
    def rvs(
        self,
        /,
        alpha: _ArrayLike_uif_1d,
        size: tuple[int, int],
        random_state: spt.Seed | None = None,
    ) -> onpt.Array[tuple[int, int, int], np.float64]: ...
    @overload
    def rvs(
        self,
        /,
        alpha: _ArrayLike_uif_1d,
        size: onpt.AtLeast2D,
        random_state: spt.Seed | None = None,
    ) -> onpt.Array[onpt.AtLeast3D, np.float64]: ...

class dirichlet_frozen(multi_rv_frozen[dirichlet_gen]):
    alpha: Final[onpt.Array[tuple[int], _Scalar_uif]]
    def __init__(self, /, alpha: _ArrayLike_uif_1d, seed: spt.Seed | None = None) -> None: ...
    def logpdf(self, /, x: _ArrayLike_uif_nd) -> _ScalarOrArray_f8: ...
    def pdf(self, /, x: _ArrayLike_uif_nd) -> _ScalarOrArray_f8: ...
    def mean(self, /) -> onpt.Array[tuple[int], np.float64]: ...
    def var(self, /) -> onpt.Array[tuple[int], np.float64]: ...
    def cov(self, /) -> onpt.Array[tuple[int, int], np.float64]: ...
    def entropy(self, /) -> np.float64: ...
    @overload
    def rvs(self, /, size: tuple[()], random_state: spt.Seed | None = None) -> onpt.Array[tuple[int], np.float64]: ...
    @overload
    def rvs(
        self,
        /,
        size: int | tuple[int] = 1,
        random_state: spt.Seed | None = None,
    ) -> onpt.Array[tuple[int, int], np.float64]: ...
    @overload
    def rvs(
        self,
        /,
        size: tuple[int, int],
        random_state: spt.Seed | None = None,
    ) -> onpt.Array[tuple[int, int, int], np.float64]: ...
    @overload
    def rvs(self, /, size: onpt.AtLeast2D, random_state: spt.Seed | None = None) -> onpt.Array[onpt.AtLeast3D, np.float64]: ...

class wishart_gen(multi_rv_generic):
    @override
    def __call__(
        self,
        /,
        df: spt.AnyReal | None = None,
        scale: _ArrayLike_uif_2d_max | None = None,
        seed: spt.Seed | None = None,
    ) -> wishart_frozen: ...
    def logpdf(
        self,
        /,
        x: _ArrayLike_uif_nd,
        df: spt.AnyReal,
        scale: _ArrayLike_uif_2d_max,
    ) -> _ScalarOrArray_f8: ...
    def pdf(
        self,
        /,
        x: _ArrayLike_uif_nd,
        df: spt.AnyReal,
        scale: _ArrayLike_uif_2d_max,
    ) -> _ScalarOrArray_f8: ...
    def mean(self, /, df: spt.AnyReal, scale: _ArrayLike_uif_2d_max) -> np.float64 | onpt.Array[tuple[int, int], np.float64]: ...
    def mode(self, /, df: spt.AnyReal, scale: _ArrayLike_uif_2d_max) -> np.float64 | None: ...
    def var(self, /, df: spt.AnyReal, scale: _ArrayLike_uif_2d_max) -> np.float64 | onpt.Array[tuple[int, int], np.float64]: ...
    def rvs(
        self,
        /,
        df: spt.AnyReal,
        scale: _ArrayLike_uif_2d_max,
        size: int | tuple[int, ...] = 1,
        random_state: spt.Seed | None = None,
    ) -> _ScalarOrArray_f8: ...
    def entropy(self, /, df: spt.AnyReal, scale: _ArrayLike_uif_2d_max) -> np.float64: ...

class wishart_frozen(multi_rv_frozen[wishart_gen]):
    dim: Final[int]
    df: Final[spt.AnyReal]
    scale: Final[onpt.Array[tuple[int, int], np.float64]]
    C: Final[onpt.Array[tuple[int, int], np.float64]]
    log_det_scale: Final[float]
    def __init__(self, df: spt.AnyReal, scale: _ArrayLike_uif_2d_max, seed: spt.Seed | None = None) -> None: ...
    def logpdf(self, /, x: _ArrayLike_uif_nd) -> _ScalarOrArray_f8: ...
    def pdf(self, /, x: _ArrayLike_uif_nd) -> _ScalarOrArray_f8: ...
    def mean(self, /) -> np.float64 | onpt.Array[tuple[int, int], np.float64]: ...
    def mode(self, /) -> np.float64 | None: ...
    def var(self, /) -> np.float64 | onpt.Array[tuple[int, int], np.float64]: ...
    def rvs(
        self,
        /,
        size: int | onpt.AtLeast1D = 1,
        random_state: spt.Seed | None = None,
    ) -> _ScalarOrArray_f8: ...
    def entropy(self, /) -> np.float64: ...

class invwishart_gen(wishart_gen):
    @override
    def __call__(  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        /,
        df: spt.AnyReal | None = None,
        scale: _ArrayLike_uif_2d_max | None = None,
        seed: spt.Seed | None = None,
    ) -> invwishart_frozen: ...
    @override
    def mean(self, /, df: spt.AnyReal, scale: _ArrayLike_uif_2d_max) -> np.float64 | None: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def mode(self, /, df: spt.AnyReal, scale: _ArrayLike_uif_2d_max) -> np.float64 | onpt.Array[tuple[int, int], np.float64]: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def var(self, /, df: spt.AnyReal, scale: _ArrayLike_uif_2d_max) -> np.float64 | None: ...  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]

class invwishart_frozen(multi_rv_frozen[invwishart_gen]):
    def __init__(self, /, df: spt.AnyReal, scale: _ArrayLike_uif_2d_max, seed: spt.Seed | None = None) -> None: ...
    def logpdf(self, /, x: _ArrayLike_uif_nd) -> _ScalarOrArray_f8: ...
    def pdf(self, /, x: _ArrayLike_uif_nd) -> _ScalarOrArray_f8: ...
    def mean(self, /) -> np.float64 | onpt.Array[tuple[int, int], np.float64]: ...
    def mode(self, /) -> np.float64 | None: ...
    def var(self, /) -> np.float64 | onpt.Array[tuple[int, int], np.float64]: ...
    def rvs(self, /, size: int | tuple[int, ...] = 1, random_state: spt.Seed | None = None) -> npt.NDArray[np.float64]: ...
    def entropy(self, /) -> np.float64: ...

# NOTE: `n` and `p` are broadcast-able (although this breaks `.rvs()` at runtime...)
class multinomial_gen(multi_rv_generic):
    @override
    def __call__(self, /, n: onpt.AnyIntegerArray, p: _ArrayLike_f_nd, seed: spt.Seed | None = None) -> multinomial_frozen: ...
    def logpmf(self, /, x: _ArrayLike_uif_nd, n: onpt.AnyIntegerArray, p: _ArrayLike_f_nd) -> _ScalarOrArray_f8: ...
    def pmf(self, /, x: _ArrayLike_uif_nd, n: onpt.AnyIntegerArray, p: _ArrayLike_f_nd) -> _ScalarOrArray_f8: ...
    def mean(self, /, n: onpt.AnyIntegerArray, p: _ArrayLike_f_nd) -> onpt.Array[onpt.AtLeast1D, np.float64]: ...
    def cov(self, /, n: onpt.AnyIntegerArray, p: _ArrayLike_f_nd) -> onpt.Array[onpt.AtLeast2D, np.float64]: ...
    def entropy(self, /, n: onpt.AnyIntegerArray, p: _ArrayLike_f_nd) -> _ScalarOrArray_f8: ...
    @overload
    def rvs(
        self,
        /,
        n: onpt.AnyIntegerArray,
        p: _ArrayLike_f_nd,
        size: tuple[()],
        random_state: spt.Seed | None = None,
    ) -> onpt.Array[onpt.AtLeast1D]: ...
    @overload
    def rvs(
        self,
        /,
        n: onpt.AnyIntegerArray,
        p: _ArrayLike_f_nd,
        size: int | onpt.AtLeast1D | None = None,
        random_state: spt.Seed | None = None,
    ) -> onpt.Array[onpt.AtLeast2D]: ...

# TODO: make generic on the (output) shape-type, i.e. to capture the broadcasting effects (use `TypeVar` with default)
class multinomial_frozen(multi_rv_frozen[multinomial_gen]):
    def __init__(self, /, n: onpt.AnyIntegerArray, p: _ArrayLike_f_nd, seed: spt.Seed | None = None) -> None: ...
    def logpmf(self, /, x: _ArrayLike_uif_nd) -> _ScalarOrArray_f8: ...
    def pmf(self, /, x: _ArrayLike_uif_nd) -> _ScalarOrArray_f8: ...
    def mean(self, /) -> onpt.Array[onpt.AtLeast1D, np.float64]: ...
    def cov(self, /) -> onpt.Array[onpt.AtLeast2D, np.float64]: ...
    def entropy(self, /) -> _ScalarOrArray_f8: ...
    @overload
    def rvs(self, /, size: tuple[()], random_state: spt.Seed | None = None) -> onpt.Array[onpt.AtLeast1D, np.float64]: ...
    @overload
    def rvs(
        self,
        /,
        size: onpt.AtLeast1D | int = 1,
        random_state: spt.Seed | None = None,
    ) -> onpt.Array[onpt.AtLeast2D, np.float64]: ...

@type_check_only
class _group_rv_gen_mixin(Generic[_RVF_co]):
    def __call__(self, /, dim: spt.AnyInt | None = None, seed: spt.Seed | None = None) -> _RVF_co: ...
    def rvs(
        self,
        /,
        dim: spt.AnyInt,
        size: int = 1,
        random_state: spt.Seed | None = None,
    ) -> onpt.Array[tuple[int, int, int], np.float64]: ...

@type_check_only
class _group_rv_frozen_mixin:
    dim: spt.AnyInt
    def __init__(self, /, dim: spt.AnyInt | None = None, seed: spt.Seed | None = None) -> None: ...
    def rvs(self, /, size: int = 1, random_state: spt.Seed | None = None) -> onpt.Array[tuple[int, int, int], np.float64]: ...

class special_ortho_group_gen(_group_rv_gen_mixin[special_ortho_group_frozen], multi_rv_generic): ...
class special_ortho_group_frozen(_group_rv_frozen_mixin, multi_rv_frozen[special_ortho_group_gen]): ...
class ortho_group_gen(_group_rv_gen_mixin[ortho_group_frozen], multi_rv_generic): ...
class ortho_group_frozen(_group_rv_frozen_mixin, multi_rv_frozen[ortho_group_gen]): ...
class unitary_group_gen(_group_rv_gen_mixin[unitary_group_frozen], multi_rv_generic): ...
class unitary_group_frozen(_group_rv_frozen_mixin, multi_rv_frozen[unitary_group_gen]): ...

# TODO: `uniform_direction` is vector-valued => make mixins generic on the shape-type, default to `tuple[int, int, int]`
class uniform_direction_gen(_group_rv_gen_mixin[uniform_direction_frozen], multi_rv_generic): ...
class uniform_direction_frozen(_group_rv_frozen_mixin, multi_rv_frozen[uniform_direction_gen]): ...

class random_correlation_gen(multi_rv_generic):
    @override
    def __call__(
        self,
        /,
        eigs: _ArrayLike_uif_1d,
        seed: spt.Seed | None = None,
        tol: float = 1e-13,
        diag_tol: float = 1e-07,
    ) -> random_correlation_frozen: ...
    def rvs(
        self,
        /,
        eigs: _ArrayLike_uif_1d,
        random_state: spt.Seed | None = None,
        tol: float = 1e-13,
        diag_tol: float = 1e-07,
    ) -> npt.NDArray[np.float64]: ...

class random_correlation_frozen(multi_rv_frozen[random_correlation_gen]):
    tol: Final[float]
    diag_tol: Final[float]
    eigs: Final[onpt.Array[tuple[int], np.float64]]
    def __init__(
        self,
        /,
        eigs: _ArrayLike_uif_1d,
        seed: spt.Seed | None = None,
        tol: float = 1e-13,
        diag_tol: float = 1e-07,
    ) -> None: ...
    def rvs(self, /, random_state: spt.Seed | None = None) -> npt.NDArray[np.float64]: ...

class multivariate_t_gen(multi_rv_generic):
    @override
    def __call__(
        self,
        /,
        loc: _ArrayLike_uif_1d | None = None,
        shape: spt.AnyReal | _ArrayLike_uif_2d = 1,
        df: int = 1,
        allow_singular: bool = False,
        seed: spt.Seed | None = None,
    ) -> multivariate_t_frozen: ...
    def pdf(
        self,
        /,
        x: _ArrayLike_uif_nd,
        loc: _ArrayLike_uif_1d | None = None,
        shape: spt.AnyReal | _ArrayLike_uif_2d = 1,
        df: int = 1,
        allow_singular: bool = False,
    ) -> _ScalarOrArray_f8: ...
    def logpdf(
        self,
        /,
        x: _ArrayLike_uif_nd,
        loc: _ArrayLike_uif_1d | None = None,
        shape: spt.AnyReal | _ArrayLike_uif_2d = 1,
        df: int = 1,
    ) -> _ScalarOrArray_f8: ...
    def cdf(
        self,
        /,
        x: _ArrayLike_uif_nd,
        loc: _ArrayLike_uif_1d | None = None,
        shape: spt.AnyReal | _ArrayLike_uif_2d = 1,
        df: int = 1,
        allow_singular: bool = False,
        *,
        maxpts: int | None = None,
        lower_limit: _ArrayLike_uif_1d | None = None,
        random_state: spt.Seed | None = None,
    ) -> _ScalarOrArray_f8: ...
    def entropy(
        self,
        /,
        loc: _ArrayLike_uif_1d | None = None,
        shape: spt.AnyReal | _ArrayLike_uif_2d = 1,
        df: int = 1,
    ) -> np.float64: ...
    @overload
    def rvs(
        self,
        /,
        loc: _ArrayLike_uif_1d | None = None,
        shape: spt.AnyReal | _ArrayLike_uif_2d = 1,
        df: int = 1,
        *,
        size: tuple[()],
        random_state: spt.Seed | None = None,
    ) -> onpt.Array[tuple[int], np.float64]: ...
    @overload
    def rvs(
        self,
        loc: _ArrayLike_uif_1d | None,
        shape: spt.AnyReal | _ArrayLike_uif_2d,
        df: int,
        /,
        size: tuple[()],
        random_state: spt.Seed | None = None,
    ) -> onpt.Array[tuple[int], np.float64]: ...
    @overload
    def rvs(
        self,
        /,
        loc: _ArrayLike_uif_1d | None = None,
        shape: spt.AnyReal | _ArrayLike_uif_2d = 1,
        df: int = 1,
        size: int | tuple[int] = 1,
        random_state: spt.Seed | None = None,
    ) -> onpt.Array[tuple[int, int], np.float64]: ...
    @overload
    def rvs(
        self,
        /,
        loc: _ArrayLike_uif_1d | None = None,
        shape: spt.AnyReal | _ArrayLike_uif_2d = 1,
        df: int = 1,
        *,
        size: onpt.AtLeast2D,
        random_state: spt.Seed | None = None,
    ) -> onpt.Array[onpt.AtLeast3D, np.float64]: ...
    @overload
    def rvs(
        self,
        loc: _ArrayLike_uif_1d | None,
        shape: spt.AnyReal | _ArrayLike_uif_2d,
        df: int,
        /,
        size: onpt.AtLeast2D,
        random_state: spt.Seed | None = None,
    ) -> onpt.Array[onpt.AtLeast3D, np.float64]: ...

class multivariate_t_frozen(multi_rv_frozen[multivariate_t_gen]):
    dim: Final[int]
    loc: Final[onpt.Array[tuple[int], np.float64]]
    shape: Final[onpt.Array[tuple[int, int], np.float64]]
    df: Final[int]
    shape_info: Final[_covariance._PSD]
    def __init__(
        self,
        /,
        loc: _ArrayLike_uif_1d | None = None,
        shape: spt.AnyReal | _ArrayLike_uif_2d = 1,
        df: int = 1,
        allow_singular: bool = False,
        seed: spt.Seed | None = None,
    ) -> None: ...
    def logpdf(self, /, x: _ArrayLike_uif_nd) -> _ScalarOrArray_f8: ...
    def pdf(self, /, x: _ArrayLike_uif_nd) -> _ScalarOrArray_f8: ...
    def cdf(
        self,
        /,
        x: _ArrayLike_uif_nd,
        *,
        maxpts: int | None = None,
        lower_limit: _ArrayLike_uif_1d | None = None,
        random_state: spt.Seed | None = None,
    ) -> _ScalarOrArray_f8: ...
    def entropy(self, /) -> np.float64: ...
    @overload
    def rvs(self, /, size: tuple[()], random_state: spt.Seed | None = None) -> onpt.Array[tuple[int], np.float64]: ...
    @overload
    def rvs(
        self,
        /,
        size: int | tuple[int] = 1,
        random_state: spt.Seed | None = None,
    ) -> onpt.Array[tuple[int, int], np.float64]: ...
    @overload
    def rvs(self, /, size: onpt.AtLeast2D, random_state: spt.Seed | None = None) -> onpt.Array[onpt.AtLeast3D, np.float64]: ...

# NOTE: `m` and `n` are broadcastable (but doing so will break `.rvs()` at runtime...)
class multivariate_hypergeom_gen(multi_rv_generic):
    @override
    def __call__(
        self,
        /,
        m: _ArrayLike_ui_nd,
        n: int | _ArrayLike_ui_nd,
        seed: spt.Seed | None = None,
    ) -> multivariate_hypergeom_frozen: ...
    def logpmf(self, /, x: _ArrayLike_uif_nd, m: _ArrayLike_ui_nd, n: int | _ArrayLike_ui_nd) -> _ScalarOrArray_f8: ...
    def pmf(self, /, x: _ArrayLike_uif_nd, m: _ArrayLike_ui_nd, n: int | _ArrayLike_ui_nd) -> _ScalarOrArray_f8: ...
    def mean(self, /, m: _ArrayLike_ui_nd, n: int | _ArrayLike_ui_nd) -> onpt.Array[onpt.AtLeast1D, np.float64]: ...
    def var(self, /, m: _ArrayLike_ui_nd, n: int | _ArrayLike_ui_nd) -> onpt.Array[onpt.AtLeast1D, np.float64]: ...
    def cov(self, /, m: _ArrayLike_ui_nd, n: int | _ArrayLike_ui_nd) -> onpt.Array[onpt.AtLeast2D, np.float64]: ...
    @overload
    def rvs(
        self,
        /,
        m: _ArrayLike_ui_nd,
        n: int | _ArrayLike_ui_nd,
        size: tuple[()],
        random_state: spt.Seed | None = None,
    ) -> onpt.Array[onpt.AtLeast1D, np.float64]: ...
    @overload
    def rvs(
        self,
        /,
        m: _ArrayLike_ui_nd,
        n: int | _ArrayLike_ui_nd,
        size: int | onpt.AtLeast1D | None = None,
        random_state: spt.Seed | None = None,
    ) -> onpt.Array[onpt.AtLeast2D, np.float64]: ...

# TODO: make generic with a shape type-param (with default) to capture the broadcasting effects
class multivariate_hypergeom_frozen(multi_rv_frozen[multivariate_hypergeom_gen]):
    def __init__(self, /, m: _ArrayLike_ui_nd, n: int | _ArrayLike_ui_nd, seed: spt.Seed | None = None) -> None: ...
    def logpmf(self, /, x: _ArrayLike_uif_nd) -> _ScalarOrArray_f8: ...
    def pmf(self, /, x: _ArrayLike_uif_nd) -> _ScalarOrArray_f8: ...
    def mean(self, /) -> onpt.Array[onpt.AtLeast1D, np.float64]: ...
    def var(self, /) -> onpt.Array[onpt.AtLeast1D, np.float64]: ...
    def cov(self, /) -> onpt.Array[onpt.AtLeast2D, np.float64]: ...
    @overload
    def rvs(self, /, size: tuple[()], random_state: spt.Seed | None = None) -> onpt.Array[onpt.AtLeast1D, np.float64]: ...
    @overload
    def rvs(
        self,
        /,
        size: int | onpt.AtLeast1D = 1,
        random_state: spt.Seed | None = None,
    ) -> onpt.Array[onpt.AtLeast2D, np.float64]: ...

_RandomTableRVSMethod: TypeAlias = Literal["boyett", "patefield"]

class random_table_gen(multi_rv_generic):
    @override
    def __call__(
        self,
        /,
        row: _ArrayLike_ui_nd,
        col: _ArrayLike_ui_nd,
        *,
        seed: spt.Seed | None = None,
    ) -> random_table_frozen: ...
    def logpmf(self, /, x: _ArrayLike_uif_nd, row: _ArrayLike_ui_nd, col: _ArrayLike_ui_nd) -> _ScalarOrArray_f8: ...
    def pmf(self, /, x: _ArrayLike_uif_nd, row: _ArrayLike_ui_nd, col: _ArrayLike_ui_nd) -> _ScalarOrArray_f8: ...
    def mean(self, /, row: _ArrayLike_ui_nd, col: _ArrayLike_ui_nd) -> onpt.Array[tuple[int, int], np.float64]: ...
    def rvs(
        self,
        /,
        row: _ArrayLike_ui_nd,
        col: _ArrayLike_ui_nd,
        *,
        size: int | None = None,
        method: _RandomTableRVSMethod | None = None,
        random_state: spt.Seed | None = None,
    ) -> onpt.Array[tuple[int, int, int], np.float64]: ...

class random_table_frozen(multi_rv_frozen[random_table_gen]):
    def __init__(self, /, row: _ArrayLike_ui_nd, col: _ArrayLike_ui_nd, *, seed: spt.Seed | None = None) -> None: ...
    def logpmf(self, /, x: _ArrayLike_uif_nd) -> _ScalarOrArray_f8: ...
    def pmf(self, /, x: _ArrayLike_uif_nd) -> _ScalarOrArray_f8: ...
    def mean(self, /) -> onpt.Array[tuple[int, int], np.float64]: ...
    def rvs(
        self,
        /,
        size: int | None = None,
        method: _RandomTableRVSMethod | None = None,
        random_state: spt.Seed | None = None,
    ) -> onpt.Array[tuple[int, int, int], np.float64]: ...

class dirichlet_multinomial_gen(multi_rv_generic):
    @override
    def __call__(
        self,
        /,
        alpha: _ArrayLike_uif_nd,
        n: onpt.AnyIntegerArray,
        seed: spt.Seed | None = None,
    ) -> dirichlet_multinomial_frozen: ...
    def logpmf(self, /, x: _ArrayLike_ui_nd, alpha: _ArrayLike_uif_nd, n: onpt.AnyIntegerArray) -> _ScalarOrArray_f8: ...
    def pmf(self, /, x: _ArrayLike_ui_nd, alpha: _ArrayLike_uif_nd, n: onpt.AnyIntegerArray) -> _ScalarOrArray_f8: ...
    def mean(self, /, alpha: _ArrayLike_uif_nd, n: onpt.AnyIntegerArray) -> onpt.Array[onpt.AtLeast1D, np.float64]: ...
    def var(self, /, alpha: _ArrayLike_uif_nd, n: onpt.AnyIntegerArray) -> onpt.Array[onpt.AtLeast1D, np.float64]: ...
    def cov(self, /, alpha: _ArrayLike_uif_nd, n: onpt.AnyIntegerArray) -> onpt.Array[onpt.AtLeast2D, np.float64]: ...

# TODO: make generic with a shape type-param (with default) to capture the broadcasting effects
class dirichlet_multinomial_frozen(multi_rv_frozen[dirichlet_multinomial_gen]):
    alpha: onpt.Array[onpt.AtLeast1D, np.float64]
    n: onpt.Array[onpt.AtLeast1D, np.int_]  # broadcasted against alpha
    def __init__(self, /, alpha: _ArrayLike_uif_nd, n: onpt.AnyIntegerArray, seed: spt.Seed | None = None) -> None: ...
    def logpmf(self, /, x: _ArrayLike_ui_nd) -> _ScalarOrArray_f8: ...
    def pmf(self, /, x: _ArrayLike_ui_nd) -> _ScalarOrArray_f8: ...
    def mean(self, /) -> onpt.Array[onpt.AtLeast1D, np.float64]: ...
    def var(self, /) -> onpt.Array[onpt.AtLeast1D, np.float64]: ...
    def cov(self, /) -> onpt.Array[onpt.AtLeast2D, np.float64]: ...

class vonmises_fisher_gen(multi_rv_generic):
    @override
    def __call__(
        self,
        /,
        mu: _ArrayLike_uif_1d | None = None,
        kappa: int = 1,
        seed: spt.Seed | None = None,
    ) -> vonmises_fisher_frozen: ...
    def logpdf(self, /, x: _ArrayLike_uif_nd, mu: _ArrayLike_uif_1d | None = None, kappa: int = 1) -> _ScalarOrArray_f8: ...
    def pdf(self, /, x: _ArrayLike_uif_nd, mu: _ArrayLike_uif_1d | None = None, kappa: int = 1) -> _ScalarOrArray_f8: ...
    def entropy(self, /, mu: _ArrayLike_uif_1d | None = None, kappa: int = 1) -> np.float64: ...
    def rvs(
        self,
        /,
        mu: _ArrayLike_uif_1d | None = None,
        kappa: int = 1,
        size: int | onpt.AtLeast1D = 1,
        random_state: spt.Seed | None = None,
    ) -> onpt.Array[onpt.AtLeast2D, np.float64]: ...
    def fit(self, /, x: _ArrayLike_uif_nd) -> tuple[onpt.Array[tuple[int], np.float64], float]: ...

class vonmises_fisher_frozen(multi_rv_frozen[vonmises_fisher_gen]):
    def __init__(self, /, mu: _ArrayLike_uif_1d | None = None, kappa: int = 1, seed: spt.Seed | None = None) -> None: ...
    def logpdf(self, /, x: _ArrayLike_uif_nd) -> _ScalarOrArray_f8: ...
    def pdf(self, /, x: _ArrayLike_uif_nd) -> _ScalarOrArray_f8: ...
    def entropy(self, /) -> np.float64: ...
    def rvs(
        self,
        /,
        size: int | onpt.AtLeast1D = 1,
        random_state: spt.Seed | None = None,
    ) -> onpt.Array[onpt.AtLeast2D, np.float64]: ...

multivariate_normal: Final[multivariate_normal_gen]
matrix_normal: Final[matrix_normal_gen]
dirichlet: Final[dirichlet_gen]
wishart: Final[wishart_gen]
invwishart: Final[invwishart_gen]
multinomial: Final[multinomial_gen]
special_ortho_group: Final[special_ortho_group_gen]
ortho_group: Final[ortho_group_gen]
random_correlation: Final[random_correlation_gen]
unitary_group: Final[unitary_group_gen]
multivariate_t: Final[multivariate_t_gen]
multivariate_hypergeom: Final[multivariate_hypergeom_gen]
random_table: Final[random_table_gen]
uniform_direction: Final[uniform_direction_gen]
dirichlet_multinomial: Final[dirichlet_multinomial_gen]
vonmises_fisher: Final[vonmises_fisher_gen]
