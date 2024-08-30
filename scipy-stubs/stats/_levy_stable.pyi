from collections.abc import Callable
from typing import Final

import numpy as np
import numpy.typing as npt
from ._distn_infrastructure import rv_continuous

__all__ = ["levy_stable", "levy_stable_gen", "pdf_from_cf_with_fft"]

class levy_stable_gen(rv_continuous):
    parameterization: Final = "S1"
    pdf_default_method: Final = "piecewise"
    cdf_default_method: Final = "piecewise"
    quad_eps: Final = 1.2e-14
    piecewise_x_tol_near_zeta: Final = 0.005
    piecewise_alpha_tol_near_one: Final = 0.005
    pdf_fft_min_points_threshold: Final = None
    pdf_fft_grid_spacing: Final = 0.001
    pdf_fft_n_points_two_power: Final = None
    pdf_fft_interpolation_level: Final = 3
    pdf_fft_interpolation_degree: Final = 3

levy_stable: Final[levy_stable_gen]

def pdf_from_cf_with_fft(
    cf: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.complex128]],
    h: float = 0.01,
    q: int = 9,
    level: int = 3,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
