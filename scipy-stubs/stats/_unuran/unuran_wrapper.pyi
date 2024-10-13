from collections.abc import Callable
from typing import NamedTuple, Protocol, overload, type_check_only

import numpy as np
import numpy.typing as npt
import scipy.stats as stats
from scipy._typing import AnyReal, Seed

__all__ = ["DiscreteAliasUrn", "NumericalInversePolynomial", "TransformedDensityRejection", "UNURANError"]

@type_check_only
class _HasSupport(Protocol):
    @property
    def support(self) -> tuple[float, float]: ...

@type_check_only
class _HasPMF(_HasSupport, Protocol):
    @property
    def pmf(self) -> Callable[..., float]: ...

@type_check_only
class _HasPDF(_HasSupport, Protocol):
    @property
    def pdf(self) -> Callable[..., float]: ...

@type_check_only
class _HasCDF(_HasPDF, Protocol):
    @property
    def cdf(self) -> Callable[..., float]: ...

@type_check_only
class _TDRDist(_HasPDF, Protocol):
    @property
    def dpdf(self) -> Callable[..., float]: ...

@type_check_only
class _PINVDist(_HasCDF, Protocol):
    @property
    def logpdf(self) -> Callable[..., float]: ...

@type_check_only
class _PPFMethodMixin:
    @overload
    def ppf(self, u: AnyReal) -> float: ...
    @overload
    def ppf(self, u: npt.ArrayLike) -> float | npt.NDArray[np.float64]: ...

class UNURANError(RuntimeError): ...

class UError(NamedTuple):
    max_error: float
    mean_absolute_error: float

class Method:
    @overload
    def rvs(self, size: None = None, random_state: Seed | None = None) -> float | int: ...
    @overload
    def rvs(self, size: int | tuple[int, ...]) -> npt.NDArray[np.float64 | np.int_]: ...
    def set_random_state(self, random_state: Seed | None = None) -> None: ...

class TransformedDensityRejection(Method):
    def __init__(
        self,
        dist: _TDRDist,
        *,
        mode: float | None = ...,
        center: float | None = ...,
        domain: tuple[float, float] | None = ...,
        c: float = ...,
        construction_points: npt.ArrayLike = ...,
        use_dars: bool = ...,
        max_squeeze_hat_ratio: float = ...,
        random_state: Seed | None = ...,
    ) -> None: ...
    @property
    def hat_area(self) -> float: ...
    @property
    def squeeze_hat_ratio(self) -> float: ...
    @property
    def squeeze_area(self) -> float: ...
    @overload
    def ppf_hat(self, u: AnyReal) -> float: ...
    @overload
    def ppf_hat(self, u: npt.ArrayLike) -> float | npt.NDArray[np.float64]: ...

class SimpleRatioUniforms(Method):
    def __init__(
        self,
        dist: _HasPDF,
        *,
        mode: float | None = ...,
        pdf_area: float = ...,
        domain: tuple[float, float] | None = ...,
        cdf_at_mode: float = ...,
        random_state: Seed | None = ...,
    ) -> None: ...

class NumericalInversePolynomial(_PPFMethodMixin, Method):
    def __init__(
        self,
        dist: _PINVDist,
        *,
        mode: float | None = ...,
        center: float | None = ...,
        domain: tuple[float, float] | None = ...,
        order: int = ...,
        u_resolution: float = ...,
        random_state: Seed | None = ...,
    ) -> None: ...
    @property
    def intervals(self) -> int: ...
    @overload
    def cdf(self, x: AnyReal) -> float: ...
    @overload
    def cdf(self, x: npt.ArrayLike) -> float | npt.NDArray[np.float64]: ...
    def u_error(self, sample_size: int = ...) -> UError: ...
    def qrvs(
        self,
        size: int | tuple[int, ...] | None = ...,
        d: int | None = ...,
        qmc_engine: stats.qmc.QMCEngine | None = ...,
    ) -> npt.ArrayLike: ...

class NumericalInverseHermite(_PPFMethodMixin, Method):
    def __init__(
        self,
        dist: _HasCDF,
        *,
        domain: tuple[float, float] | None = ...,
        order: int = ...,
        u_resolution: float = ...,
        construction_points: npt.ArrayLike | None = ...,
        max_intervals: int = ...,
        random_state: Seed | None = ...,
    ) -> None: ...
    @property
    def intervals(self) -> int: ...
    @property
    def midpoint_error(self) -> float: ...
    def u_error(self, sample_size: int = ...) -> UError: ...
    def qrvs(
        self,
        size: int | tuple[int, ...] | None = ...,
        d: int | None = ...,
        qmc_engine: stats.qmc.QMCEngine | None = ...,
    ) -> npt.ArrayLike: ...

class DiscreteAliasUrn(Method):
    def __init__(
        self,
        dist: npt.ArrayLike | _HasPMF,
        *,
        domain: tuple[float, float] | None = ...,
        urn_factor: float = ...,
        random_state: Seed | None = ...,
    ) -> None: ...

class DiscreteGuideTable(_PPFMethodMixin, Method):
    def __init__(
        self,
        dist: npt.ArrayLike | _HasPMF,
        *,
        domain: tuple[float, float] | None = ...,
        guide_factor: float = ...,
        random_state: Seed | None = ...,
    ) -> None: ...
