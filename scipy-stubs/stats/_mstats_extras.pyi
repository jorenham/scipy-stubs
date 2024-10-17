from scipy._typing import Untyped

__all__ = [
    "compare_medians_ms",
    "hdmedian",
    "hdquantiles",
    "hdquantiles_sd",
    "idealfourths",
    "median_cihs",
    "mjci",
    "mquantiles_cimj",
    "rsh",
    "trimmed_mean_ci",
]

def hdquantiles(data: Untyped, prob: Untyped = [0.25, 0.5, 0.75], axis: int | None = None, var: bool = False) -> Untyped: ...
def hdmedian(data: Untyped, axis: int = -1, var: bool = False) -> Untyped: ...
def hdquantiles_sd(data: Untyped, prob: Untyped = [0.25, 0.5, 0.75], axis: int | None = None) -> Untyped: ...
def trimmed_mean_ci(
    data: Untyped,
    limits: Untyped = (0.2, 0.2),
    inclusive: Untyped = (True, True),
    alpha: float = 0.05,
    axis: int | None = None,
) -> Untyped: ...
def mjci(data: Untyped, prob: Untyped = [0.25, 0.5, 0.75], axis: int | None = None) -> Untyped: ...
def mquantiles_cimj(
    data: Untyped,
    prob: Untyped = [0.25, 0.5, 0.75],
    alpha: float = 0.05,
    axis: int | None = None,
) -> Untyped: ...
def median_cihs(data: Untyped, alpha: float = 0.05, axis: int | None = None) -> Untyped: ...
def compare_medians_ms(group_1: Untyped, group_2: Untyped, axis: int | None = None) -> Untyped: ...
def idealfourths(data: Untyped, axis: int | None = None) -> Untyped: ...
def rsh(data: Untyped, points: Untyped | None = None) -> Untyped: ...
