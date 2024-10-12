from scipy._typing import Untyped
from scipy.stats.distributions import rv_continuous

__all__ = ["levy_stable", "levy_stable_gen", "pdf_from_cf_with_fft"]

Cotes: Untyped
Cotes_table: Untyped

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

levy_stable: levy_stable_gen

def pdf_from_cf_with_fft(cf: Untyped, h: float = 0.01, q: int = 9, level: int = 3) -> Untyped: ...
