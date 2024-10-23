from collections.abc import Callable, Sequence
from typing import Any, Concatenate, Final, Generic, Literal, overload
from typing_extensions import TypeVar

import numpy as np
import numpy.typing as npt
from numpy._typing import _ArrayLikeFloat_co
from optype import CanFloat
from scipy._typing import AnyReal, AnyShape, Seed
from ._distn_infrastructure import rv_frozen
from .qmc import QMCEngine

__all__ = ["FastGeneratorInversion", "RatioUniforms"]

_RT_co = TypeVar("_RT_co", covariant=True, bound=AnyReal, default=AnyReal)

###

PINV_CONFIG: Final[dict[str, dict[str, Callable[..., AnyReal]]]]

class CustomDistPINV(Generic[_RT_co]):  # undocumented
    def __init__(self, /, pdf: Callable[Concatenate[AnyReal, ...], _RT_co], args: Sequence[object]) -> None: ...
    def pdf(self, /, x: AnyReal) -> _RT_co: ...

class RatioUniforms:
    def __init__(
        self,
        /,
        pdf: Callable[Concatenate[AnyReal, ...], AnyReal],
        *,
        umax: AnyReal,
        vmin: AnyReal,
        vmax: AnyReal,
        c: AnyReal = 0,
        random_state: Seed | None = None,
    ) -> None: ...
    def rvs(self, /, size: AnyShape = 1) -> np.float64 | npt.NDArray[np.float64]: ...

class FastGeneratorInversion:
    def __init__(
        self,
        /,
        dist: rv_frozen[Any, float | np.float64],
        *,
        domain: tuple[AnyReal, AnyReal] | None = None,
        ignore_shape_range: bool = False,
        random_state: Seed | None = None,
    ) -> None: ...
    @property
    def random_state(self, /) -> np.random.Generator: ...
    @random_state.setter
    def random_state(self, random_state: Seed, /) -> None: ...
    @property
    def loc(self, /) -> float | np.float64: ...
    @loc.setter
    def loc(self, loc: AnyReal, /) -> None: ...
    @property
    def scale(self, /) -> float | np.float64: ...
    @scale.setter
    def scale(self, scale: AnyReal, /) -> None: ...
    @overload
    def rvs(self, /, size: None = None) -> np.float64: ...
    @overload
    def rvs(self, /, size: AnyShape) -> npt.NDArray[np.float64]: ...
    @overload
    def qrvs(
        self,
        /,
        size: None | tuple[Literal[1]] = None,
        d: int | None = None,
        qmc_engine: QMCEngine | None = None,
    ) -> np.float64: ...
    @overload
    def qrvs(
        self,
        /,
        size: AnyShape,
        d: int | None = None,
        qmc_engine: QMCEngine | None = None,
    ) -> np.float64 | npt.NDArray[np.float64]: ...
    def ppf(self, q: _ArrayLikeFloat_co) -> npt.NDArray[np.float64]: ...
    def evaluate_error(
        self,
        /,
        size: int = 100_000,
        random_state: Seed | None = None,
        x_error: bool = False,
    ) -> tuple[np.float64, np.float64]: ...
    def support(self, /) -> tuple[float, float] | tuple[np.float64, np.float64]: ...

def argus_pdf(x: AnyReal, chi: AnyReal) -> float: ...  # undocumented
def argus_gamma_trf(x: AnyReal, chi: AnyReal) -> np.float64: ...  # undocumented
def argus_gamma_inv_trf(x: AnyReal, chi: AnyReal) -> AnyReal: ...  # undocumented
def betaprime_pdf(x: AnyReal, a: AnyReal, b: AnyReal) -> float | np.float64: ...  # undocumented
def beta_valid_params(a: CanFloat, b: CanFloat) -> bool: ...  # undocumented
def gamma_pdf(x: AnyReal, a: CanFloat) -> float: ...  # undocumented
def invgamma_pdf(x: AnyReal, a: CanFloat) -> float: ...  # undocumented
def burr_pdf(x: AnyReal, cc: CanFloat, dd: CanFloat) -> np.float64 | Literal[0]: ...  # undocumented
def burr12_pdf(x: AnyReal, cc: AnyReal, dd: AnyReal) -> float: ...  # undocumented
def chi_pdf(x: AnyReal, a: AnyReal) -> float: ...  # undocumented
def chi2_pdf(x: AnyReal, df: AnyReal) -> float: ...  # undocumented
def alpha_pdf(x: AnyReal, a: AnyReal) -> float: ...  # undocumented
def bradford_pdf(x: AnyReal, c: AnyReal) -> AnyReal: ...  # undocumented
def crystalball_pdf(x: AnyReal, b: AnyReal, m: AnyReal) -> float: ...  # undocumented
def weibull_min_pdf(x: AnyReal, c: AnyReal) -> AnyReal: ...  # undocumented
def weibull_max_pdf(x: AnyReal, c: AnyReal) -> AnyReal: ...  # undocumented
def invweibull_pdf(x: AnyReal, c: AnyReal) -> AnyReal: ...  # undocumented
def wald_pdf(x: AnyReal) -> float: ...  # undocumented
def geninvgauss_mode(p: CanFloat, b: AnyReal) -> AnyReal: ...  # undocumented
def geninvgauss_pdf(x: AnyReal, p: AnyReal, b: AnyReal) -> float: ...  # undocumented
def invgauss_mode(mu: AnyReal) -> float: ...  # undocumented
def invgauss_pdf(x: AnyReal, mu: AnyReal) -> float: ...  # undocumented
def powerlaw_pdf(x: AnyReal, a: AnyReal) -> AnyReal: ...  # undocumented
