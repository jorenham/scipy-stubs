# mypy: disable-error-code="explicit-override"

import abc
from collections.abc import Iterable
from typing import Any, Generic, Literal as L, TypeAlias, overload, type_check_only
from typing_extensions import TypeVar

import numpy as np
import optype as op
import optype.numpy as onp
from scipy._typing import ToRNG
from ._qmc import QMCEngine

_T = TypeVar("_T")
_XT = TypeVar("_XT", bound=np.number[Any], default=np.number[Any])
_XT_co = TypeVar("_XT_co", bound=np.number[Any], default=np.float64, covariant=True)
_ShapeT = TypeVar("_ShapeT", bound=onp.AtLeast1D, default=onp.AtLeast1D)
_ShapeT0 = TypeVar("_ShapeT0", bound=tuple[int, ...], default=tuple[int, ...])
_ShapeT0_co = TypeVar("_ShapeT0_co", bound=tuple[int, ...], default=tuple[int, ...], covariant=True)

_Tuple2: TypeAlias = tuple[_T, _T]
_ToQRNG: TypeAlias = QMCEngine | ToRNG

_MedianMethod: TypeAlias = L["formula", "icdf"] | None
_ModeMethod: TypeAlias = L["formula", "optimization"] | None
_SampleMethod: TypeAlias = L["formula", "inverse_transform"] | None
_RMomentMethod: TypeAlias = L["formula", "transform", "quadrature", "cache"] | None
_CMomentMethod: TypeAlias = L["formula", "transform", "quadrature", "cache", "normalize"] | None
_SMomentMethod: TypeAlias = L["formula", "transform", "general", "cache", "normalize"] | None
_KurtosisConvention: TypeAlias = L["non-excess", "excess"]
_EntropyMethod: TypeAlias = L["formula", "logexp", "quadrature"] | None

_PDFMethod: TypeAlias = L["formula", "logexp"] | None
_CDFMethod: TypeAlias = L["formula", "logexp", "complement", "quadrature", "subtraction"] | None
_CCDFMethod: TypeAlias = L["formula", "logexp", "complement", "quadrature", "addition"] | None
_ICDFMethod: TypeAlias = L["formula", "complement", "inversion"] | None

_Float: TypeAlias = np.float64 | np.longdouble
_Float1D: TypeAlias = onp.Array1D[_Float]
_Float2D: TypeAlias = onp.Array2D[_Float]
_Float3D: TypeAlias = onp.Array3D[_Float]
_FloatND: TypeAlias = onp.ArrayND[_Float, _ShapeT]

_Complex: TypeAlias = np.complex128 | np.clongdouble
_ComplexND: TypeAlias = onp.ArrayND[_Complex, _ShapeT]

_ToFloatND: TypeAlias = onp.CanArrayND[np.floating[Any] | np.integer[Any] | np.bool_, _ShapeT]
_ToFloat0ND: TypeAlias = onp.ToFloat | onp.ToFloatND
_ToFloatMax1D: TypeAlias = onp.ToFloatStrict1D | onp.ToFloat
_ToFloatMax2D: TypeAlias = onp.ToFloatStrict2D | _ToFloatMax1D
_ToFloatMax3D: TypeAlias = onp.ToFloatStrict3D | _ToFloatMax2D
_ToFloatMaxND: TypeAlias = _ToFloatND[_ShapeT] | _ToFloatMax1D

###

class _ProbabilityDistribution(Generic[_XT_co], metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def support(self, /) -> _Tuple2[_XT_co | onp.ArrayND[_XT_co]]: ...
    @abc.abstractmethod
    def median(self, /, *, method: _MedianMethod) -> _XT_co | onp.ArrayND[_XT_co]: ...
    @abc.abstractmethod
    def mode(self, /, *, method: _ModeMethod) -> _XT_co | onp.ArrayND[_XT_co]: ...
    @abc.abstractmethod
    def sample(
        self,
        /,
        shape: int | tuple[int, ...],
        *,
        method: _SampleMethod,
        rng: _ToQRNG,
    ) -> _XT_co | onp.Array[Any, _XT_co]: ...  # `Any` shape is needed on `numpy<2.1`

    #
    @abc.abstractmethod
    def mean(self, /, *, method: _RMomentMethod) -> _XT_co | onp.ArrayND[_XT_co]: ...
    @abc.abstractmethod
    def variance(self, /, *, method: _CMomentMethod) -> _XT_co | onp.ArrayND[_XT_co]: ...
    @abc.abstractmethod
    def standard_deviation(self, /, *, method: _CMomentMethod) -> _XT_co | onp.ArrayND[_XT_co]: ...
    @abc.abstractmethod
    def skewness(self, /, *, method: _SMomentMethod) -> _XT_co | onp.ArrayND[_XT_co]: ...
    @abc.abstractmethod
    def kurtosis(self, /, *, method: _SMomentMethod) -> _XT_co | onp.ArrayND[_XT_co]: ...

    #
    @overload
    @abc.abstractmethod
    def moment(self, /, order: onp.ToInt, kind: L["raw"], *, method: _RMomentMethod) -> _Float | _FloatND: ...
    @overload
    @abc.abstractmethod
    def moment(self, /, order: onp.ToInt, kind: L["central"], *, method: _CMomentMethod) -> _Float | _FloatND: ...
    @overload
    @abc.abstractmethod
    def moment(self, /, order: onp.ToInt, kind: L["standardized"], *, method: _SMomentMethod) -> _Float | _FloatND: ...

    #
    @abc.abstractmethod
    def entropy(self, /, *, method: _EntropyMethod) -> _Float | _FloatND: ...
    @abc.abstractmethod
    def logentropy(self, /, *, method: _EntropyMethod) -> _Complex | _ComplexND: ...

    #
    @abc.abstractmethod
    def pdf(self, x: _ToFloat0ND, /, *, method: _PDFMethod) -> _Float | _FloatND: ...
    @abc.abstractmethod
    def logpdf(self, x: _ToFloat0ND, /, *, method: _PDFMethod) -> _Float | _FloatND: ...

    #
    @abc.abstractmethod
    def cdf(self, x: _ToFloat0ND, y: _ToFloat0ND | None, /, *, method: _CDFMethod) -> _Float | _FloatND: ...
    @abc.abstractmethod
    def icdf(self, p: _ToFloat0ND, /, *, method: _ICDFMethod) -> _Float | _FloatND: ...
    @abc.abstractmethod
    def ccdf(self, x: _ToFloat0ND, y: _ToFloat0ND | None, /, *, method: _CCDFMethod) -> _Float | _FloatND: ...
    @abc.abstractmethod
    def iccdf(self, p: _ToFloat0ND, /, *, method: _ICDFMethod) -> _Float | _FloatND: ...
    @abc.abstractmethod
    def logcdf(self, x: _ToFloat0ND, y: _ToFloat0ND | None, /, *, method: _CDFMethod) -> _Float | _FloatND: ...
    @abc.abstractmethod
    def ilogcdf(self, logp: _ToFloat0ND, /, *, method: _ICDFMethod) -> _Float | _FloatND: ...
    @abc.abstractmethod
    def logccdf(self, x: _ToFloat0ND, y: _ToFloat0ND | None, /, *, method: _CCDFMethod) -> _Float | _FloatND: ...
    @abc.abstractmethod
    def ilogccdf(self, logp: _ToFloat0ND, /, *, method: _ICDFMethod) -> _Float | _FloatND: ...

###

_Self: TypeAlias = _BaseDistribution[_XT, _ShapeT0]
_Self0: TypeAlias = _Self[_XT, tuple[()]]
_Self1: TypeAlias = _Self[_XT, tuple[int]]
_Self2: TypeAlias = _Self[_XT, tuple[int, int]]
_Self3: TypeAlias = _Self[_XT, tuple[int, int, int]]
_Self1_: TypeAlias = _Self[_XT, onp.AtLeast1D]

# TODO(jorenham): Merge into ContinuousDistribution?
# NOTE: the incompatible method overrides appear to be pyright-only false positives
@type_check_only
class _BaseDistribution(_ProbabilityDistribution[_XT_co], Generic[_XT_co, _ShapeT0_co]):
    @overload
    def support(self: _Self0[_XT], /) -> _Tuple2[_XT]: ...
    @overload
    def support(self: _Self[_XT, _ShapeT], /) -> _Tuple2[onp.Array[_ShapeT, _XT]]: ...  # pyright: ignore[reportIncompatibleMethodOverride]

    #
    @overload
    def median(self: _Self0[_XT], /, *, method: _MedianMethod = None) -> _XT: ...
    @overload
    def median(self: _Self[_XT, _ShapeT], /, *, method: _MedianMethod = None) -> onp.Array[_ShapeT, _XT]: ...  # pyright: ignore[reportIncompatibleMethodOverride]

    #
    @overload
    def mode(self: _Self0[_XT], /, *, method: _ModeMethod = None) -> _XT: ...
    @overload
    def mode(self: _Self[_XT, _ShapeT], /, *, method: _ModeMethod = None) -> onp.Array[_ShapeT, _XT]: ...  # pyright: ignore[reportIncompatibleMethodOverride]

    #
    @overload
    def sample(self: _Self0[_XT], /, shape: tuple[()] = (), *, method: _SampleMethod = None, rng: _ToQRNG = None) -> _XT: ...
    @overload
    def sample(
        self: _Self0[_XT],
        /,
        shape: op.CanIndex,
        *,
        method: _SampleMethod = None,
        rng: _ToQRNG = None,
    ) -> onp.Array1D[_XT]: ...
    @overload
    def sample(
        self: _Self0[_XT],
        /,
        shape: _ShapeT,
        *,
        method: _SampleMethod = None,
        rng: _ToQRNG = None,
    ) -> onp.ArrayND[_XT, _ShapeT]: ...
    @overload
    def sample(
        self: _Self[_XT, _ShapeT],
        /,
        shape: tuple[()] = (),
        *,
        method: _SampleMethod = None,
        rng: _ToQRNG = None,
    ) -> onp.ArrayND[_XT, _ShapeT]: ...
    @overload
    def sample(
        self: _Self[_XT, _ShapeT],
        /,
        shape: op.CanIndex | Iterable[op.CanIndex],
        *,
        method: _SampleMethod = None,
        rng: _ToQRNG = None,
    ) -> onp.ArrayND[_XT, _ShapeT] | onp.ArrayND[_XT]: ...  # first union type is needed on `numpy<2.1`
    @overload
    def sample(
        self,
        /,
        shape: op.CanIndex | Iterable[op.CanIndex],
        *,
        method: _SampleMethod = None,
        rng: _ToQRNG = None,
    ) -> _XT_co | onp.ArrayND[_XT_co, _ShapeT] | onp.ArrayND[_XT_co]: ...  # first union type is needed on `numpy<2.1`

    #
    @overload
    def mean(self: _Self0[_XT], /, *, method: _RMomentMethod = None) -> _XT: ...
    @overload
    def mean(self: _Self[_XT, _ShapeT], /, *, method: _RMomentMethod = None) -> onp.ArrayND[_XT, _ShapeT]: ...  # pyright: ignore[reportIncompatibleMethodOverride]

    #
    @overload
    def variance(self: _Self0[_XT], /, *, method: _CMomentMethod = None) -> _XT: ...
    @overload
    def variance(self: _Self[_XT, _ShapeT], /, *, method: _CMomentMethod = None) -> onp.ArrayND[_XT, _ShapeT]: ...  # pyright: ignore[reportIncompatibleMethodOverride]

    #
    @overload
    def standard_deviation(self: _Self0[_XT], /, *, method: _CMomentMethod = None) -> _XT: ...
    @overload
    def standard_deviation(self: _Self[_XT, _ShapeT], /, *, method: _CMomentMethod = None) -> onp.ArrayND[_XT, _ShapeT]: ...  # pyright: ignore[reportIncompatibleMethodOverride]

    #
    @overload
    def skewness(self: _Self0[_XT], /, *, method: _SMomentMethod = None) -> _XT: ...
    @overload
    def skewness(self: _Self[_XT, _ShapeT], /, *, method: _SMomentMethod = None) -> onp.ArrayND[_XT, _ShapeT]: ...  # pyright: ignore[reportIncompatibleMethodOverride]

    #
    @overload
    def kurtosis(
        self: _Self0[_XT],
        /,
        *,
        method: _SMomentMethod = None,
        convention: _KurtosisConvention = "non-excess",
    ) -> _XT: ...
    @overload
    def kurtosis(  # pyright: ignore[reportIncompatibleMethodOverride]
        self: _Self[_XT, _ShapeT],
        /,
        *,
        method: _SMomentMethod = None,
        convention: _KurtosisConvention = "non-excess",
    ) -> onp.ArrayND[_XT, _ShapeT]: ...

    #
    @overload
    def moment(self: _Self0, /, order: onp.ToInt = 1, kind: L["raw"] = "raw", *, method: _RMomentMethod = None) -> _Float: ...
    @overload
    def moment(self: _Self0, /, order: onp.ToInt, kind: L["central"], *, method: _CMomentMethod = None) -> _Float: ...
    @overload
    def moment(self: _Self0, /, order: onp.ToInt = 1, *, kind: L["central"], method: _CMomentMethod = None) -> _Float: ...
    @overload
    def moment(self: _Self0, /, order: onp.ToInt, kind: L["standardized"], *, method: _SMomentMethod = None) -> _Float: ...
    @overload
    def moment(self: _Self0, /, order: onp.ToInt = 1, *, kind: L["standardized"], method: _SMomentMethod = None) -> _Float: ...
    @overload
    def moment(
        self: _Self[Any, _ShapeT],
        /,
        order: onp.ToInt = 1,
        kind: L["raw"] = "raw",
        *,
        method: _RMomentMethod = None,
    ) -> _FloatND[_ShapeT]: ...
    @overload
    def moment(
        self: _Self[Any, _ShapeT],
        /,
        order: onp.ToInt,
        kind: L["central"],
        *,
        method: _CMomentMethod = None,
    ) -> _FloatND[_ShapeT]: ...
    @overload
    def moment(
        self: _Self[Any, _ShapeT],
        /,
        order: onp.ToInt = 1,
        *,
        kind: L["central"],
        method: _CMomentMethod = None,
    ) -> _FloatND[_ShapeT]: ...
    @overload
    def moment(
        self: _Self[Any, _ShapeT],
        /,
        order: onp.ToInt,
        kind: L["standardized"],
        *,
        method: _SMomentMethod = None,
    ) -> _FloatND[_ShapeT]: ...
    @overload
    def moment(  # pyright: ignore[reportIncompatibleMethodOverride]
        self: _Self[Any, _ShapeT],
        /,
        order: onp.ToInt = 1,
        *,
        kind: L["standardized"],
        method: _SMomentMethod = None,
    ) -> _FloatND[_ShapeT]: ...

    #
    @overload
    def entropy(self: _Self0, /, *, method: _EntropyMethod = None) -> _Float: ...
    @overload
    def entropy(self: _Self[Any, _ShapeT], /, *, method: _EntropyMethod = None) -> _FloatND[_ShapeT]: ...

    #
    @overload
    def logentropy(self: _Self0, /, *, method: _EntropyMethod = None) -> _Complex: ...
    @overload
    def logentropy(self: _Self[Any, _ShapeT], /, *, method: _EntropyMethod = None) -> _ComplexND[_ShapeT]: ...

    #
    # TODO(jorenham): Adjust these, depending on the result of https://github.com/scipy/scipy/issues/22145
    # NOTE: The signatures of `pdf` and `logpdf` are equivalent
    @overload  # self: T1-d, x: 0-d
    def pdf(self: _Self[Any, _ShapeT], x: onp.ToFloat, /, *, method: _PDFMethod = None) -> _FloatND[_ShapeT]: ...
    @overload  # self: 0-d, x: 0-d
    def pdf(self: _Self0, x: onp.ToFloat, /, *, method: _PDFMethod = None) -> _Float: ...
    @overload  # self: 0-d, x: 1-d
    def pdf(self: _Self0, x: onp.ToFloatStrict1D, /, *, method: _PDFMethod = None) -> _Float1D: ...
    @overload  # self: 0-d, x: 2-d
    def pdf(self: _Self0, x: onp.ToFloatStrict2D, /, *, method: _PDFMethod = None) -> _Float2D: ...
    @overload  # self: 0-d, x: 3-d
    def pdf(self: _Self0, x: onp.ToFloatStrict3D, /, *, method: _PDFMethod = None) -> _Float3D: ...
    @overload  # self: 0-d, x: T1-d
    def pdf(self: _Self0, x: _ToFloatND[_ShapeT], /, *, method: _PDFMethod = None) -> _FloatND[_ShapeT]: ...
    @overload  # self: 0-d, x: >=1-d
    def pdf(  # first union type is needed on `numpy<2.1`
        self: _Self0,
        x: _ToFloatND[_ShapeT] | onp.ToFloatND,
        /,
        *,
        method: _PDFMethod = None,
    ) -> _FloatND[_ShapeT] | _FloatND[onp.AtLeast1D]: ...
    @overload  # self: 1-d, x: 1-d
    def pdf(self: _Self1, x: onp.ToFloatStrict1D, /, *, method: _PDFMethod = None) -> _Float2D: ...
    @overload  # self: 1-d, x: 2-d
    def pdf(self: _Self1, x: onp.ToFloatStrict2D, /, *, method: _PDFMethod = None) -> _Float3D: ...
    @overload  # self: 1-d, x: >=-d
    def pdf(self: _Self1, x: onp.ToFloatND, /, *, method: _PDFMethod = None) -> _FloatND[onp.AtLeast2D]: ...
    @overload  # self: 2-d, x: 1-d
    def pdf(self: _Self2, x: onp.ToFloatStrict1D, /, *, method: _PDFMethod = None) -> _Float3D: ...
    @overload  # self: 2-d, x: >=1-d
    def pdf(self: _Self2, x: onp.ToFloatND, /, *, method: _PDFMethod = None) -> _FloatND[onp.AtLeast3D]: ...
    @overload  # self: 3-d, x: >=1-d
    def pdf(self: _Self3, x: onp.ToFloatND, /, *, method: _PDFMethod = None) -> _FloatND[onp.AtLeast3D]: ...
    @overload  # self: >=1-d
    def pdf(self: _Self1_, x: _ToFloat0ND, /, *, method: _PDFMethod = None) -> _FloatND: ...

    #
    @overload  # self: T1-d, x: 0-d
    def logpdf(self: _Self[Any, _ShapeT], x: onp.ToFloat, /, *, method: _PDFMethod = None) -> _FloatND[_ShapeT]: ...
    @overload  # self: 0-d, x: 0-d
    def logpdf(self: _Self0, x: onp.ToFloat, /, *, method: _PDFMethod = None) -> _Float: ...
    @overload  # self: 0-d, x: 1-d
    def logpdf(self: _Self0, x: onp.ToFloatStrict1D, /, *, method: _PDFMethod = None) -> _Float1D: ...
    @overload  # self: 0-d, x: 2-d
    def logpdf(self: _Self0, x: onp.ToFloatStrict2D, /, *, method: _PDFMethod = None) -> _Float2D: ...
    @overload  # self: 0-d, x: 3-d
    def logpdf(self: _Self0, x: onp.ToFloatStrict3D, /, *, method: _PDFMethod = None) -> _Float3D: ...
    @overload  # self: 0-d, x: T1-d
    def logpdf(self: _Self0, x: _ToFloatND[_ShapeT], /, *, method: _PDFMethod = None) -> _FloatND[_ShapeT]: ...
    @overload  # self: 0-d, x: >=1-d
    def logpdf(  # first union type is needed on `numpy<2.1`
        self: _Self0,
        x: _ToFloatND[_ShapeT] | onp.ToFloatND,
        /,
        *,
        method: _PDFMethod = None,
    ) -> _FloatND[_ShapeT] | _FloatND[onp.AtLeast1D]: ...
    @overload  # self: 1-d, x: 1-d
    def logpdf(self: _Self1, x: onp.ToFloatStrict1D, /, *, method: _PDFMethod = None) -> _Float2D: ...
    @overload  # self: 1-d, x: 2-d
    def logpdf(self: _Self1, x: onp.ToFloatStrict2D, /, *, method: _PDFMethod = None) -> _Float3D: ...
    @overload  # self: 1-d, x: >=1-d
    def logpdf(self: _Self1, x: onp.ToFloatND, /, *, method: _PDFMethod = None) -> _FloatND[onp.AtLeast2D]: ...
    @overload  # self: 2-d, x: 1-d
    def logpdf(self: _Self2, x: onp.ToFloatStrict1D, /, *, method: _PDFMethod = None) -> _Float3D: ...
    @overload  # self: 2-d, x: >=1-d
    def logpdf(self: _Self2, x: onp.ToFloatND, /, *, method: _PDFMethod = None) -> _FloatND[onp.AtLeast3D]: ...
    @overload  # self: 3-d, x: >=1-d
    def logpdf(self: _Self3, x: onp.ToFloatND, /, *, method: _PDFMethod = None) -> _FloatND[onp.AtLeast3D]: ...
    @overload  # self: >=1-d
    def logpdf(self: _Self1_, x: _ToFloat0ND, /, *, method: _PDFMethod = None) -> _FloatND: ...

    #
    # NOTE: Apart from the `method` type, the signatures of `[log]cdf` and `[log]ccdf` are equivalent
    @overload  # self: T1-d, x: 0-d, y?: 0-d
    def cdf(
        self: _Self[Any, _ShapeT],
        x: onp.ToFloat,
        y: onp.ToFloat | None = None,
        /,
        *,
        method: _CDFMethod = None,
    ) -> _FloatND[_ShapeT]: ...
    @overload  # self: 0-d, x: 0-d, y?: 0-d
    def cdf(self: _Self0, x: onp.ToFloat, y: onp.ToFloat | None = None, /, *, method: _CDFMethod = None) -> _Float: ...
    @overload  # self: 0-d, x: 1-d, y?: <=1-d
    def cdf(
        self: _Self0,
        x: onp.ToFloatStrict1D,
        y: _ToFloatMax1D | None = None,
        /,
        *,
        method: _CDFMethod = None,
    ) -> _Float1D: ...
    @overload  # self: 0-d, x: <=1-d, y: 1-d
    def cdf(self: _Self0, x: _ToFloatMax1D, y: onp.ToFloatStrict1D, /, *, method: _CDFMethod = None) -> _Float1D: ...
    @overload  # self: 0-d, x: 2-d, y?: <=2-d
    def cdf(
        self: _Self0,
        x: onp.ToFloatStrict2D,
        y: _ToFloatMax2D | None = None,
        /,
        *,
        method: _CDFMethod = None,
    ) -> _Float2D: ...
    @overload  # self: 0-d, x: <=2-d, y: 2-d
    def cdf(self: _Self0, x: _ToFloatMax2D, y: onp.ToFloatStrict2D, /, *, method: _CDFMethod = None) -> _Float2D: ...
    @overload  # self: 0-d, x: 3-d, y?: <=3-d
    def cdf(
        self: _Self0,
        x: onp.ToFloatStrict3D,
        y: _ToFloatMax3D | None = None,
        /,
        *,
        method: _CDFMethod = None,
    ) -> _Float3D: ...
    @overload  # self: 0-d, x: <=3-d, y: 3-d
    def cdf(self: _Self0, x: _ToFloatMax3D, y: onp.ToFloatStrict3D, /, *, method: _CDFMethod = None) -> _Float3D: ...
    @overload  # self: 0-d, x: T1-d, y?: T1-d | <=1-d
    def cdf(
        self: _Self0,
        x: _ToFloatND[_ShapeT],
        y: _ToFloatMax1D | None = None,
        /,
        *,
        method: _CDFMethod = None,
    ) -> _FloatND[_ShapeT]: ...
    @overload  # self: 0-d, x: T1-d | <=1-d, y: T1-d
    def cdf(
        self: _Self0,
        x: _ToFloatMaxND[_ShapeT],
        y: _ToFloatND[_ShapeT],
        /,
        *,
        method: _CDFMethod = None,
    ) -> _FloatND[_ShapeT]: ...
    @overload  # self: 0-d, x: >=1-d, y?: >=0-d
    def cdf(  # first union type is needed on `numpy<2.1`
        self: _Self0,
        x: _ToFloatND[_ShapeT] | onp.ToFloatND,
        y: _ToFloat0ND | None = None,
        /,
        *,
        method: _CDFMethod = None,
    ) -> _FloatND[_ShapeT] | _FloatND[onp.AtLeast1D]: ...
    @overload  # self: 0-d, x: >=0-d, y: >=1-d
    def cdf(  # first union type is needed on `numpy<2.1`
        self: _Self0,
        x: _ToFloat0ND,
        y: _ToFloatND[_ShapeT] | onp.ToFloatND,
        /,
        *,
        method: _CDFMethod = None,
    ) -> _FloatND[_ShapeT] | _FloatND[onp.AtLeast1D]: ...
    @overload  # self: 1-d, x: 1-d, y?: <=1-d
    def cdf(
        self: _Self1,
        x: onp.ToFloatStrict1D,
        y: _ToFloatMax1D | None = None,
        /,
        *,
        method: _CDFMethod = None,
    ) -> _Float2D: ...
    @overload  # self: 1-d, x: <=1-d, y: 1-d
    def cdf(self: _Self1, x: _ToFloatMax1D, y: onp.ToFloatStrict1D, /, *, method: _CDFMethod = None) -> _Float2D: ...
    @overload  # self: 1-d, x: 2-d, y?: <=2-d
    def cdf(
        self: _Self1,
        x: onp.ToFloatStrict2D,
        y: _ToFloatMax2D | None = None,
        /,
        *,
        method: _CDFMethod = None,
    ) -> _Float3D: ...
    @overload  # self: 1-d, x: <=2-d, y: 2-d
    def cdf(self: _Self1, x: _ToFloatMax2D, y: onp.ToFloatStrict2D, /, *, method: _CDFMethod = None) -> _Float3D: ...
    @overload  # self: 1-d, x: >=1-d, y?: >=0-d
    def cdf(
        self: _Self1,
        x: onp.ToFloatND,
        y: _ToFloat0ND | None = None,
        /,
        *,
        method: _CDFMethod = None,
    ) -> _FloatND[onp.AtLeast2D]: ...
    @overload  # self: 1-d, x: >=0-d, y: >=1-d
    def cdf(self: _Self1, x: _ToFloat0ND, y: onp.ToFloatND, /, *, method: _CDFMethod = None) -> _FloatND[onp.AtLeast2D]: ...
    @overload  # self: 2-d, x: 1-d, y?: <=1-d
    def cdf(
        self: _Self2,
        x: onp.ToFloatStrict1D,
        y: _ToFloatMax1D | None = None,
        /,
        *,
        method: _CDFMethod = None,
    ) -> _Float3D: ...
    @overload  # self: 2-d, x: <=1-d, y: 1-d
    def cdf(self: _Self2, x: _ToFloatMax1D, y: onp.ToFloatStrict1D, /, *, method: _CDFMethod = None) -> _Float3D: ...
    @overload  # self: 2-d, x: >=1-d, y?: >=0-d
    def cdf(
        self: _Self2,
        x: onp.ToFloatND,
        y: _ToFloat0ND | None = None,
        /,
        *,
        method: _CDFMethod = None,
    ) -> _FloatND[onp.AtLeast3D]: ...
    @overload  # self: 2-d, x: >=0-d, y: >=1-d
    def cdf(self: _Self2, x: _ToFloat0ND, y: onp.ToFloatND, /, *, method: _CDFMethod = None) -> _FloatND[onp.AtLeast3D]: ...
    @overload  # self: 3-d, x: >=0-d, y?: >=0-d
    def cdf(
        self: _Self3,
        x: _ToFloat0ND,
        y: _ToFloat0ND | None = None,
        /,
        *,
        method: _CDFMethod = None,
    ) -> _FloatND[onp.AtLeast3D]: ...
    @overload  # self: >=1-d, x: >=0-d, y?: >=0-d
    def cdf(self: _Self1_, x: _ToFloat0ND, y: _ToFloat0ND | None = None, /, *, method: _CDFMethod = None) -> _FloatND: ...

    #
    @overload  # self: T1-d, x: 0-d, y?: 0-d
    def logcdf(
        self: _Self[Any, _ShapeT],
        x: onp.ToFloat,
        y: onp.ToFloat | None = None,
        /,
        *,
        method: _CDFMethod = None,
    ) -> _FloatND[_ShapeT]: ...
    @overload  # self: 0-d, x: 0-d, y?: 0-d
    def logcdf(self: _Self0, x: onp.ToFloat, y: onp.ToFloat | None = None, /, *, method: _CDFMethod = None) -> _Float: ...
    @overload  # self: 0-d, x: 1-d, y?: <=1-d
    def logcdf(
        self: _Self0,
        x: onp.ToFloatStrict1D,
        y: _ToFloatMax1D | None = None,
        /,
        *,
        method: _CDFMethod = None,
    ) -> _Float1D: ...
    @overload  # self: 0-d, x: <=1-d, y: 1-d
    def logcdf(self: _Self0, x: _ToFloatMax1D, y: onp.ToFloatStrict1D, /, *, method: _CDFMethod = None) -> _Float1D: ...
    @overload  # self: 0-d, x: 2-d, y?: <=2-d
    def logcdf(
        self: _Self0,
        x: onp.ToFloatStrict2D,
        y: _ToFloatMax2D | None = None,
        /,
        *,
        method: _CDFMethod = None,
    ) -> _Float2D: ...
    @overload  # self: 0-d, x: <=2-d, y: 2-d
    def logcdf(self: _Self0, x: _ToFloatMax2D, y: onp.ToFloatStrict2D, /, *, method: _CDFMethod = None) -> _Float2D: ...
    @overload  # self: 0-d, x: 3-d, y?: <=3-d
    def logcdf(
        self: _Self0,
        x: onp.ToFloatStrict3D,
        y: _ToFloatMax3D | None = None,
        /,
        *,
        method: _CDFMethod = None,
    ) -> _Float3D: ...
    @overload  # self: 0-d, x: <=3-d, y: 3-d
    def logcdf(self: _Self0, x: _ToFloatMax3D, y: onp.ToFloatStrict3D, /, *, method: _CDFMethod = None) -> _Float3D: ...
    @overload  # self: 0-d, x: T1-d, y?: T1-d | <=1-d
    def logcdf(
        self: _Self0,
        x: _ToFloatND[_ShapeT],
        y: _ToFloatMax1D | None = None,
        /,
        *,
        method: _CDFMethod = None,
    ) -> _FloatND[_ShapeT]: ...
    @overload  # self: 0-d, x: T1-d | <=1-d, y: T1-d
    def logcdf(
        self: _Self0,
        x: _ToFloatMaxND[_ShapeT],
        y: _ToFloatND[_ShapeT],
        /,
        *,
        method: _CDFMethod = None,
    ) -> _FloatND[_ShapeT]: ...
    @overload  # self: 0-d, x: >=1-d, y?: >=0-d
    def logcdf(  # first union type is needed on `numpy<2.1`
        self: _Self0,
        x: _ToFloatND[_ShapeT] | onp.ToFloatND,
        y: _ToFloat0ND | None = None,
        /,
        *,
        method: _CDFMethod = None,
    ) -> _FloatND[_ShapeT] | _FloatND[onp.AtLeast1D]: ...
    @overload  # self: 0-d, x: >=0-d, y: >=1-d
    def logcdf(  # first union type is needed on `numpy<2.1`
        self: _Self0,
        x: _ToFloat0ND,
        y: _ToFloatND[_ShapeT] | onp.ToFloatND,
        /,
        *,
        method: _CDFMethod = None,
    ) -> _FloatND[_ShapeT] | _FloatND[onp.AtLeast1D]: ...
    @overload  # self: 1-d, x: 1-d, y?: <=1-d
    def logcdf(
        self: _Self1,
        x: onp.ToFloatStrict1D,
        y: _ToFloatMax1D | None = None,
        /,
        *,
        method: _CDFMethod = None,
    ) -> _Float2D: ...
    @overload  # self: 1-d, x: <=1-d, y: 1-d
    def logcdf(self: _Self1, x: _ToFloatMax1D, y: onp.ToFloatStrict1D, /, *, method: _CDFMethod = None) -> _Float2D: ...
    @overload  # self: 1-d, x: 2-d, y?: <=2-d
    def logcdf(
        self: _Self1,
        x: onp.ToFloatStrict2D,
        y: _ToFloatMax2D | None = None,
        /,
        *,
        method: _CDFMethod = None,
    ) -> _Float3D: ...
    @overload  # self: 1-d, x: <=2-d, y: 2-d
    def logcdf(self: _Self1, x: _ToFloatMax2D, y: onp.ToFloatStrict2D, /, *, method: _CDFMethod = None) -> _Float3D: ...
    @overload  # self: 1-d, x: >=1-d, y?: >=0-d
    def logcdf(
        self: _Self1,
        x: onp.ToFloatND,
        y: _ToFloat0ND | None = None,
        /,
        *,
        method: _CDFMethod = None,
    ) -> _FloatND[onp.AtLeast2D]: ...
    @overload  # self: 1-d, x: >=0-d, y: >=1-d
    def logcdf(self: _Self1, x: _ToFloat0ND, y: onp.ToFloatND, /, *, method: _CDFMethod = None) -> _FloatND[onp.AtLeast2D]: ...
    @overload  # self: 2-d, x: 1-d, y?: <=1-d
    def logcdf(
        self: _Self2,
        x: onp.ToFloatStrict1D,
        y: _ToFloatMax1D | None = None,
        /,
        *,
        method: _CDFMethod = None,
    ) -> _Float3D: ...
    @overload  # self: 2-d, x: <=1-d, y: 1-d
    def logcdf(self: _Self2, x: _ToFloatMax1D, y: onp.ToFloatStrict1D, /, *, method: _CDFMethod = None) -> _Float3D: ...
    @overload  # self: 2-d, x: >=1-d, y?: >=0-d
    def logcdf(
        self: _Self2,
        x: onp.ToFloatND,
        y: _ToFloat0ND | None = None,
        /,
        *,
        method: _CDFMethod = None,
    ) -> _FloatND[onp.AtLeast3D]: ...
    @overload  # self: 2-d, x: >=0-d, y: >=1-d
    def logcdf(self: _Self2, x: _ToFloat0ND, y: onp.ToFloatND, /, *, method: _CDFMethod = None) -> _FloatND[onp.AtLeast3D]: ...
    @overload  # self: 3-d, x: >=0-d, y?: >=0-d
    def logcdf(
        self: _Self3,
        x: _ToFloat0ND,
        y: _ToFloat0ND | None = None,
        /,
        *,
        method: _CDFMethod = None,
    ) -> _FloatND[onp.AtLeast3D]: ...
    @overload  # self: >=1-d, x: >=0-d, y?: >=0-d
    def logcdf(self: _Self1_, x: _ToFloat0ND, y: _ToFloat0ND | None = None, /, *, method: _CDFMethod = None) -> _FloatND: ...

    #
    @overload  # self: T1-d, x: 0-d, y?: 0-d
    def ccdf(
        self: _Self[Any, _ShapeT],
        x: onp.ToFloat,
        y: onp.ToFloat | None = None,
        /,
        *,
        method: _CCDFMethod = None,
    ) -> _FloatND[_ShapeT]: ...
    @overload  # self: 0-d, x: 0-d, y?: 0-d
    def ccdf(self: _Self0, x: onp.ToFloat, y: onp.ToFloat | None = None, /, *, method: _CCDFMethod = None) -> _Float: ...
    @overload  # self: 0-d, x: 1-d, y?: <=1-d
    def ccdf(
        self: _Self0,
        x: onp.ToFloatStrict1D,
        y: _ToFloatMax1D | None = None,
        /,
        *,
        method: _CCDFMethod = None,
    ) -> _Float1D: ...
    @overload  # self: 0-d, x: <=1-d, y: 1-d
    def ccdf(self: _Self0, x: _ToFloatMax1D, y: onp.ToFloatStrict1D, /, *, method: _CCDFMethod = None) -> _Float1D: ...
    @overload  # self: 0-d, x: 2-d, y?: <=2-d
    def ccdf(
        self: _Self0,
        x: onp.ToFloatStrict2D,
        y: _ToFloatMax2D | None = None,
        /,
        *,
        method: _CCDFMethod = None,
    ) -> _Float2D: ...
    @overload  # self: 0-d, x: <=2-d, y: 2-d
    def ccdf(self: _Self0, x: _ToFloatMax2D, y: onp.ToFloatStrict2D, /, *, method: _CCDFMethod = None) -> _Float2D: ...
    @overload  # self: 0-d, x: 3-d, y?: <=3-d
    def ccdf(
        self: _Self0,
        x: onp.ToFloatStrict3D,
        y: _ToFloatMax3D | None = None,
        /,
        *,
        method: _CCDFMethod = None,
    ) -> _Float3D: ...
    @overload  # self: 0-d, x: <=3-d, y: 3-d
    def ccdf(self: _Self0, x: _ToFloatMax3D, y: onp.ToFloatStrict3D, /, *, method: _CCDFMethod = None) -> _Float3D: ...
    @overload  # self: 0-d, x: T1-d, y?: T1-d | <=1-d
    def ccdf(
        self: _Self0,
        x: _ToFloatND[_ShapeT],
        y: _ToFloatMax1D | None = None,
        /,
        *,
        method: _CCDFMethod = None,
    ) -> _FloatND[_ShapeT]: ...
    @overload  # self: 0-d, x: T1-d | <=1-d, y: T1-d
    def ccdf(
        self: _Self0,
        x: _ToFloatMaxND[_ShapeT],
        y: _ToFloatND[_ShapeT],
        /,
        *,
        method: _CCDFMethod = None,
    ) -> _FloatND[_ShapeT]: ...
    @overload  # self: 0-d, x: >=1-d, y?: >=0-d
    def ccdf(  # first union type is needed on `numpy<2.1`
        self: _Self0,
        x: _ToFloatND[_ShapeT] | onp.ToFloatND,
        y: _ToFloat0ND | None = None,
        /,
        *,
        method: _CCDFMethod = None,
    ) -> _FloatND[_ShapeT] | _FloatND[onp.AtLeast1D]: ...
    @overload  # self: 0-d, x: >=0-d, y: >=1-d
    def ccdf(  # first union type is needed on `numpy<2.1`
        self: _Self0,
        x: _ToFloat0ND,
        y: _ToFloatND[_ShapeT] | onp.ToFloatND,
        /,
        *,
        method: _CCDFMethod = None,
    ) -> _FloatND[_ShapeT] | _FloatND[onp.AtLeast1D]: ...
    @overload  # self: 1-d, x: 1-d, y?: <=1-d
    def ccdf(
        self: _Self1,
        x: onp.ToFloatStrict1D,
        y: _ToFloatMax1D | None = None,
        /,
        *,
        method: _CCDFMethod = None,
    ) -> _Float2D: ...
    @overload  # self: 1-d, x: <=1-d, y: 1-d
    def ccdf(self: _Self1, x: _ToFloatMax1D, y: onp.ToFloatStrict1D, /, *, method: _CCDFMethod = None) -> _Float2D: ...
    @overload  # self: 1-d, x: 2-d, y?: <=2-d
    def ccdf(
        self: _Self1,
        x: onp.ToFloatStrict2D,
        y: _ToFloatMax2D | None = None,
        /,
        *,
        method: _CCDFMethod = None,
    ) -> _Float3D: ...
    @overload  # self: 1-d, x: <=2-d, y: 2-d
    def ccdf(self: _Self1, x: _ToFloatMax2D, y: onp.ToFloatStrict2D, /, *, method: _CCDFMethod = None) -> _Float3D: ...
    @overload  # self: 1-d, x: >=1-d, y?: >=0-d
    def ccdf(
        self: _Self1,
        x: onp.ToFloatND,
        y: _ToFloat0ND | None = None,
        /,
        *,
        method: _CCDFMethod = None,
    ) -> _FloatND[onp.AtLeast2D]: ...
    @overload  # self: 1-d, x: >=0-d, y: >=1-d
    def ccdf(self: _Self1, x: _ToFloat0ND, y: onp.ToFloatND, /, *, method: _CCDFMethod = None) -> _FloatND[onp.AtLeast2D]: ...
    @overload  # self: 2-d, x: 1-d, y?: <=1-d
    def ccdf(
        self: _Self2,
        x: onp.ToFloatStrict1D,
        y: _ToFloatMax1D | None = None,
        /,
        *,
        method: _CCDFMethod = None,
    ) -> _Float3D: ...
    @overload  # self: 2-d, x: <=1-d, y: 1-d
    def ccdf(self: _Self2, x: _ToFloatMax1D, y: onp.ToFloatStrict1D, /, *, method: _CCDFMethod = None) -> _Float3D: ...
    @overload  # self: 2-d, x: >=1-d, y?: >=0-d
    def ccdf(
        self: _Self2,
        x: onp.ToFloatND,
        y: _ToFloat0ND | None = None,
        /,
        *,
        method: _CCDFMethod = None,
    ) -> _FloatND[onp.AtLeast3D]: ...
    @overload  # self: 2-d, x: >=0-d, y: >=1-d
    def ccdf(self: _Self2, x: _ToFloat0ND, y: onp.ToFloatND, /, *, method: _CCDFMethod = None) -> _FloatND[onp.AtLeast3D]: ...
    @overload  # self: 3-d, x: >=0-d, y?: >=0-d
    def ccdf(
        self: _Self3,
        x: _ToFloat0ND,
        y: _ToFloat0ND | None = None,
        /,
        *,
        method: _CCDFMethod = None,
    ) -> _FloatND[onp.AtLeast3D]: ...
    @overload  # self: >=1-d, x: >=0-d, y?: >=0-d
    def ccdf(self: _Self1_, x: _ToFloat0ND, y: _ToFloat0ND | None = None, /, *, method: _CCDFMethod = None) -> _FloatND: ...

    #
    @overload  # self: T1-d, x: 0-d, y?: 0-d
    def logccdf(
        self: _Self[Any, _ShapeT],
        x: onp.ToFloat,
        y: onp.ToFloat | None = None,
        /,
        *,
        method: _CCDFMethod = None,
    ) -> _FloatND[_ShapeT]: ...
    @overload  # self: 0-d, x: 0-d, y?: 0-d
    def logccdf(self: _Self0, x: onp.ToFloat, y: onp.ToFloat | None = None, /, *, method: _CCDFMethod = None) -> _Float: ...
    @overload  # self: 0-d, x: 1-d, y?: <=1-d
    def logccdf(
        self: _Self0,
        x: onp.ToFloatStrict1D,
        y: _ToFloatMax1D | None = None,
        /,
        *,
        method: _CCDFMethod = None,
    ) -> _Float1D: ...
    @overload  # self: 0-d, x: <=1-d, y: 1-d
    def logccdf(self: _Self0, x: _ToFloatMax1D, y: onp.ToFloatStrict1D, /, *, method: _CCDFMethod = None) -> _Float1D: ...
    @overload  # self: 0-d, x: 2-d, y?: <=2-d
    def logccdf(
        self: _Self0,
        x: onp.ToFloatStrict2D,
        y: _ToFloatMax2D | None = None,
        /,
        *,
        method: _CCDFMethod = None,
    ) -> _Float2D: ...
    @overload  # self: 0-d, x: <=2-d, y: 2-d
    def logccdf(self: _Self0, x: _ToFloatMax2D, y: onp.ToFloatStrict2D, /, *, method: _CCDFMethod = None) -> _Float2D: ...
    @overload  # self: 0-d, x: 3-d, y?: <=3-d
    def logccdf(
        self: _Self0,
        x: onp.ToFloatStrict3D,
        y: _ToFloatMax3D | None = None,
        /,
        *,
        method: _CCDFMethod = None,
    ) -> _Float3D: ...
    @overload  # self: 0-d, x: <=3-d, y: 3-d
    def logccdf(self: _Self0, x: _ToFloatMax3D, y: onp.ToFloatStrict3D, /, *, method: _CCDFMethod = None) -> _Float3D: ...
    @overload  # self: 0-d, x: T1-d, y?: T1-d | <=1-d
    def logccdf(
        self: _Self0,
        x: _ToFloatND[_ShapeT],
        y: _ToFloatMax1D | None = None,
        /,
        *,
        method: _CCDFMethod = None,
    ) -> _FloatND[_ShapeT]: ...
    @overload  # self: 0-d, x: T1-d | <=1-d, y: T1-d
    def logccdf(
        self: _Self0,
        x: _ToFloatMaxND[_ShapeT],
        y: _ToFloatND[_ShapeT],
        /,
        *,
        method: _CCDFMethod = None,
    ) -> _FloatND[_ShapeT]: ...
    @overload  # self: 0-d, x: >=1-d, y?: >=0-d
    def logccdf(  # first union type is needed on `numpy<2.1`
        self: _Self0,
        x: _ToFloatND[_ShapeT] | onp.ToFloatND,
        y: _ToFloat0ND | None = None,
        /,
        *,
        method: _CCDFMethod = None,
    ) -> _FloatND[_ShapeT] | _FloatND[onp.AtLeast1D]: ...
    @overload  # self: 0-d, x: >=0-d, y: >=1-d
    def logccdf(  # first union type is needed on `numpy<2.1`
        self: _Self0,
        x: _ToFloat0ND,
        y: _ToFloatND[_ShapeT] | onp.ToFloatND,
        /,
        *,
        method: _CCDFMethod = None,
    ) -> _FloatND[_ShapeT] | _FloatND[onp.AtLeast1D]: ...
    @overload  # self: 1-d, x: 1-d, y?: <=1-d
    def logccdf(
        self: _Self1,
        x: onp.ToFloatStrict1D,
        y: _ToFloatMax1D | None = None,
        /,
        *,
        method: _CCDFMethod = None,
    ) -> _Float2D: ...
    @overload  # self: 1-d, x: <=1-d, y: 1-d
    def logccdf(self: _Self1, x: _ToFloatMax1D, y: onp.ToFloatStrict1D, /, *, method: _CCDFMethod = None) -> _Float2D: ...
    @overload  # self: 1-d, x: 2-d, y?: <=2-d
    def logccdf(
        self: _Self1,
        x: onp.ToFloatStrict2D,
        y: _ToFloatMax2D | None = None,
        /,
        *,
        method: _CCDFMethod = None,
    ) -> _Float3D: ...
    @overload  # self: 1-d, x: <=2-d, y: 2-d
    def logccdf(self: _Self1, x: _ToFloatMax2D, y: onp.ToFloatStrict2D, /, *, method: _CCDFMethod = None) -> _Float3D: ...
    @overload  # self: 1-d, x: >=1-d, y?: >=0-d
    def logccdf(
        self: _Self1,
        x: onp.ToFloatND,
        y: _ToFloat0ND | None = None,
        /,
        *,
        method: _CCDFMethod = None,
    ) -> _FloatND[onp.AtLeast2D]: ...
    @overload  # self: 1-d, x: >=0-d, y: >=1-d
    def logccdf(self: _Self1, x: _ToFloat0ND, y: onp.ToFloatND, /, *, method: _CCDFMethod = None) -> _FloatND[onp.AtLeast2D]: ...
    @overload  # self: 2-d, x: 1-d, y?: <=1-d
    def logccdf(
        self: _Self2,
        x: onp.ToFloatStrict1D,
        y: _ToFloatMax1D | None = None,
        /,
        *,
        method: _CCDFMethod = None,
    ) -> _Float3D: ...
    @overload  # self: 2-d, x: <=1-d, y: 1-d
    def logccdf(self: _Self2, x: _ToFloatMax1D, y: onp.ToFloatStrict1D, /, *, method: _CCDFMethod = None) -> _Float3D: ...
    @overload  # self: 2-d, x: >=1-d, y?: >=0-d
    def logccdf(
        self: _Self2,
        x: onp.ToFloatND,
        y: _ToFloat0ND | None = None,
        /,
        *,
        method: _CCDFMethod = None,
    ) -> _FloatND[onp.AtLeast3D]: ...
    @overload  # self: 2-d, x: >=0-d, y: >=1-d
    def logccdf(self: _Self2, x: _ToFloat0ND, y: onp.ToFloatND, /, *, method: _CCDFMethod = None) -> _FloatND[onp.AtLeast3D]: ...
    @overload  # self: 3-d, x: >=0-d, y?: >=0-d
    def logccdf(
        self: _Self3,
        x: _ToFloat0ND,
        y: _ToFloat0ND | None = None,
        /,
        *,
        method: _CCDFMethod = None,
    ) -> _FloatND[onp.AtLeast3D]: ...
    @overload  # self: >=1-d, x: >=0-d, y?: >=0-d
    def logccdf(self: _Self1_, x: _ToFloat0ND, y: _ToFloat0ND | None = None, /, *, method: _CCDFMethod = None) -> _FloatND: ...

    # NOTE: Apart from the `method` type, the signatures of `i[log]cdf` and `i[log]ccdf` are equivalent to those of `[log]pdf`
    @overload  # self: T1-d, p: 0-d
    def icdf(self: _Self[Any, _ShapeT], p: onp.ToFloat, /, *, method: _ICDFMethod = None) -> _FloatND[_ShapeT]: ...
    @overload  # self: 0-d, p: 0-d
    def icdf(self: _Self0, p: onp.ToFloat, /, *, method: _ICDFMethod = None) -> _Float: ...
    @overload  # self: 0-d, p: 1-d
    def icdf(self: _Self0, p: onp.ToFloatStrict1D, /, *, method: _ICDFMethod = None) -> _Float1D: ...
    @overload  # self: 0-d, p: 2-d
    def icdf(self: _Self0, p: onp.ToFloatStrict2D, /, *, method: _ICDFMethod = None) -> _Float2D: ...
    @overload  # self: 0-d, p: 3-d
    def icdf(self: _Self0, p: onp.ToFloatStrict3D, /, *, method: _ICDFMethod = None) -> _Float3D: ...
    @overload  # self: 0-d, p: T1-d
    def icdf(self: _Self0, p: _ToFloatND[_ShapeT], /, *, method: _ICDFMethod = None) -> _FloatND[_ShapeT]: ...
    @overload  # self: 0-d, p: >=1-d
    def icdf(  # first union type is needed on `numpy<2.1`
        self: _Self0,
        p: _ToFloatND[_ShapeT] | onp.ToFloatND,
        /,
        *,
        method: _ICDFMethod = None,
    ) -> _FloatND[_ShapeT] | _FloatND[onp.AtLeast1D]: ...
    @overload  # self: 1-d, p: 1-d
    def icdf(self: _Self1, p: onp.ToFloatStrict1D, /, *, method: _ICDFMethod = None) -> _Float2D: ...
    @overload  # self: 1-d, p: 2-d
    def icdf(self: _Self1, p: onp.ToFloatStrict2D, /, *, method: _ICDFMethod = None) -> _Float3D: ...
    @overload  # self: 1-d, p: >=-d
    def icdf(self: _Self1, p: onp.ToFloatND, /, *, method: _ICDFMethod = None) -> _FloatND[onp.AtLeast2D]: ...
    @overload  # self: 2-d, p: 1-d
    def icdf(self: _Self2, p: onp.ToFloatStrict1D, /, *, method: _ICDFMethod = None) -> _Float3D: ...
    @overload  # self: 2-d, p: >=1-d
    def icdf(self: _Self2, p: onp.ToFloatND, /, *, method: _ICDFMethod = None) -> _FloatND[onp.AtLeast3D]: ...
    @overload  # self: 3-d, p: >=1-d
    def icdf(self: _Self3, p: onp.ToFloatND, /, *, method: _ICDFMethod = None) -> _FloatND[onp.AtLeast3D]: ...
    @overload  # self: >=1-d
    def icdf(self: _Self1_, p: _ToFloat0ND, /, *, method: _ICDFMethod = None) -> _FloatND: ...
    #
    @overload  # self: T1-d, logp: 0-d
    def ilogcdf(self: _Self[Any, _ShapeT], logp: onp.ToFloat, /, *, method: _ICDFMethod = None) -> _FloatND[_ShapeT]: ...
    @overload  # self: 0-d, logp: 0-d
    def ilogcdf(self: _Self0, logp: onp.ToFloat, /, *, method: _ICDFMethod = None) -> _Float: ...
    @overload  # self: 0-d, logp: 1-d
    def ilogcdf(self: _Self0, logp: onp.ToFloatStrict1D, /, *, method: _ICDFMethod = None) -> _Float1D: ...
    @overload  # self: 0-d, logp: 2-d
    def ilogcdf(self: _Self0, logp: onp.ToFloatStrict2D, /, *, method: _ICDFMethod = None) -> _Float2D: ...
    @overload  # self: 0-d, logp: 3-d
    def ilogcdf(self: _Self0, logp: onp.ToFloatStrict3D, /, *, method: _ICDFMethod = None) -> _Float3D: ...
    @overload  # self: 0-d, logp: T1-d
    def ilogcdf(self: _Self0, logp: _ToFloatND[_ShapeT], /, *, method: _ICDFMethod = None) -> _FloatND[_ShapeT]: ...
    @overload  # self: 0-d, p: >=1-d
    def ilogcdf(  # first union type is needed on `numpy<2.1`
        self: _Self0,
        logp: _ToFloatND[_ShapeT] | onp.ToFloatND,
        /,
        *,
        method: _ICDFMethod = None,
    ) -> _FloatND[_ShapeT] | _FloatND[onp.AtLeast1D]: ...
    @overload  # self: 1-d, logp: 1-d
    def ilogcdf(self: _Self1, logp: onp.ToFloatStrict1D, /, *, method: _ICDFMethod = None) -> _Float2D: ...
    @overload  # self: 1-d, logp: 2-d
    def ilogcdf(self: _Self1, logp: onp.ToFloatStrict2D, /, *, method: _ICDFMethod = None) -> _Float3D: ...
    @overload  # self: 1-d, logp: >=-d
    def ilogcdf(self: _Self1, logp: onp.ToFloatND, /, *, method: _ICDFMethod = None) -> _FloatND[onp.AtLeast2D]: ...
    @overload  # self: 2-d, logp: 1-ds
    def ilogcdf(self: _Self2, logp: onp.ToFloatStrict1D, /, *, method: _ICDFMethod = None) -> _Float3D: ...
    @overload  # self: 2-d, logp: >=1-d
    def ilogcdf(self: _Self2, logp: onp.ToFloatND, /, *, method: _ICDFMethod = None) -> _FloatND[onp.AtLeast3D]: ...
    @overload  # self: 3-d, logp: >=1-d
    def ilogcdf(self: _Self3, logp: onp.ToFloatND, /, *, method: _ICDFMethod = None) -> _FloatND[onp.AtLeast3D]: ...
    @overload  # self: >=1-d
    def ilogcdf(self: _Self1_, logp: _ToFloat0ND, /, *, method: _ICDFMethod = None) -> _FloatND: ...

    #
    @overload  # self: T1-d, p: 0-d
    def iccdf(self: _Self[Any, _ShapeT], p: onp.ToFloat, /, *, method: _ICDFMethod = None) -> _FloatND[_ShapeT]: ...
    @overload  # self: 0-d, p: 0-d
    def iccdf(self: _Self0, p: onp.ToFloat, /, *, method: _ICDFMethod = None) -> _Float: ...
    @overload  # self: 0-d, p: 1-d
    def iccdf(self: _Self0, p: onp.ToFloatStrict1D, /, *, method: _ICDFMethod = None) -> _Float1D: ...
    @overload  # self: 0-d, p: 2-d
    def iccdf(self: _Self0, p: onp.ToFloatStrict2D, /, *, method: _ICDFMethod = None) -> _Float2D: ...
    @overload  # self: 0-d, p: 3-d
    def iccdf(self: _Self0, p: onp.ToFloatStrict3D, /, *, method: _ICDFMethod = None) -> _Float3D: ...
    @overload  # self: 0-d, p: T1-d
    def iccdf(self: _Self0, p: _ToFloatND[_ShapeT], /, *, method: _ICDFMethod = None) -> _FloatND[_ShapeT]: ...
    @overload  # self: 0-d, p: >=1-d
    def iccdf(  # first union type is needed on `numpy<2.1`
        self: _Self0,
        p: _ToFloatND[_ShapeT] | onp.ToFloatND,
        /,
        *,
        method: _ICDFMethod = None,
    ) -> _FloatND[_ShapeT] | _FloatND[onp.AtLeast1D]: ...
    @overload  # self: 1-d, p: 1-d
    def iccdf(self: _Self1, p: onp.ToFloatStrict1D, /, *, method: _ICDFMethod = None) -> _Float2D: ...
    @overload  # self: 1-d, p: 2-d
    def iccdf(self: _Self1, p: onp.ToFloatStrict2D, /, *, method: _ICDFMethod = None) -> _Float3D: ...
    @overload  # self: 1-d, p: >=-d
    def iccdf(self: _Self1, p: onp.ToFloatND, /, *, method: _ICDFMethod = None) -> _FloatND[onp.AtLeast2D]: ...
    @overload  # self: 2-d, p: 1-d
    def iccdf(self: _Self2, p: onp.ToFloatStrict1D, /, *, method: _ICDFMethod = None) -> _Float3D: ...
    @overload  # self: 2-d, p: >=1-d
    def iccdf(self: _Self2, p: onp.ToFloatND, /, *, method: _ICDFMethod = None) -> _FloatND[onp.AtLeast3D]: ...
    @overload  # self: 3-d, p: >=1-d
    def iccdf(self: _Self3, p: onp.ToFloatND, /, *, method: _ICDFMethod = None) -> _FloatND[onp.AtLeast3D]: ...
    @overload  # self: >=1-d
    def iccdf(self: _Self1_, p: _ToFloat0ND, /, *, method: _ICDFMethod = None) -> _FloatND: ...
    #
    @overload  # self: T1-d, logp: 0-d
    def ilogccdf(self: _Self[Any, _ShapeT], logp: onp.ToFloat, /, *, method: _ICDFMethod = None) -> _FloatND[_ShapeT]: ...
    @overload  # self: 0-d, logp: 0-d
    def ilogccdf(self: _Self0, logp: onp.ToFloat, /, *, method: _ICDFMethod = None) -> _Float: ...
    @overload  # self: 0-d, logp: 1-d
    def ilogccdf(self: _Self0, logp: onp.ToFloatStrict1D, /, *, method: _ICDFMethod = None) -> _Float1D: ...
    @overload  # self: 0-d, logp: 2-d
    def ilogccdf(self: _Self0, logp: onp.ToFloatStrict2D, /, *, method: _ICDFMethod = None) -> _Float2D: ...
    @overload  # self: 0-d, logp: 3-d
    def ilogccdf(self: _Self0, logp: onp.ToFloatStrict3D, /, *, method: _ICDFMethod = None) -> _Float3D: ...
    @overload  # self: 0-d, logp: T1-d
    def ilogccdf(self: _Self0, logp: _ToFloatND[_ShapeT], /, *, method: _ICDFMethod = None) -> _FloatND[_ShapeT]: ...
    @overload  # self: 0-d, q: >=1-d
    def ilogccdf(  # first union type is needed on `numpy<2.1`
        self: _Self0,
        logp: _ToFloatND[_ShapeT] | onp.ToFloatND,
        /,
        *,
        method: _ICDFMethod = None,
    ) -> _FloatND[_ShapeT] | _FloatND[onp.AtLeast1D]: ...
    @overload  # self: 1-d, logp: 1-d
    def ilogccdf(self: _Self1, logp: onp.ToFloatStrict1D, /, *, method: _ICDFMethod = None) -> _Float2D: ...
    @overload  # self: 1-d, logp: 2-d
    def ilogccdf(self: _Self1, logp: onp.ToFloatStrict2D, /, *, method: _ICDFMethod = None) -> _Float3D: ...
    @overload  # self: 1-d, logp: >=-d
    def ilogccdf(self: _Self1, logp: onp.ToFloatND, /, *, method: _ICDFMethod = None) -> _FloatND[onp.AtLeast2D]: ...
    @overload  # self: 2-d, logp: 1-d
    def ilogccdf(self: _Self2, logp: onp.ToFloatStrict1D, /, *, method: _ICDFMethod = None) -> _Float3D: ...
    @overload  # self: 2-d, logp: >=1-d
    def ilogccdf(self: _Self2, logp: onp.ToFloatND, /, *, method: _ICDFMethod = None) -> _FloatND[onp.AtLeast3D]: ...
    @overload  # self: 3-d, logp: >=1-d
    def ilogccdf(self: _Self3, logp: onp.ToFloatND, /, *, method: _ICDFMethod = None) -> _FloatND[onp.AtLeast3D]: ...
    @overload  # self: >=1-d
    def ilogccdf(self: _Self1_, logp: _ToFloat0ND, /, *, method: _ICDFMethod = None) -> _FloatND: ...
