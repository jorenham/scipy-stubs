from collections.abc import Callable

from ._distn_infrastructure import rv_continuous
from scipy._typing import Untyped

class levy_stable_gen(rv_continuous):
    parameterization: str
    pdf_default_method: str
    cdf_default_method: str
    quad_eps: Untyped
    piecewise_x_tol_near_zeta: float
    piecewise_alpha_tol_near_one: float
    pdf_fft_min_points_threshold: Untyped
    pdf_fft_grid_spacing: float
    pdf_fft_n_points_two_power: Untyped
    pdf_fft_interpolation_level: int
    pdf_fft_interpolation_degree: int

def pdf_from_cf_with_fft(
    cf: Callable[[float], complex], h: float = ..., q: int = ..., level: int = ...
) -> tuple[Untyped, Untyped]: ...

levy_stable: levy_stable_gen
