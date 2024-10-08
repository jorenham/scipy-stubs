# This module is not meant for public use and will be removed in SciPy v2.0.0.
from typing_extensions import deprecated

__all__ = ["gaussian_kde"]

@deprecated("will be removed in SciPy v2.0.0")
class gaussian_kde:
    @property
    def inv_cov(self) -> object: ...
    @property
    def weights(self) -> object: ...
    @property
    def neff(self) -> object: ...
    def __init__(self, dataset: object, bw_method: object = ..., weights: object = ...) -> None: ...
    def evaluate(self, points: object) -> object: ...
    def integrate_gaussian(self, mean: object, cov: object) -> object: ...
    def integrate_box_1d(self, low: object, high: object) -> object: ...
    def integrate_box(self, low_bounds: object, high_bounds: object, maxpts: object = ...) -> object: ...
    def integrate_kde(self, other: object) -> object: ...
    def resample(self, size: object = ..., seed: object = ...) -> object: ...
    def scotts_factor(self) -> object: ...
    def silverman_factor(self) -> object: ...
    def set_bandwidth(self, bw_method: object = ...) -> object: ...
    def pdf(self, x: object) -> object: ...
    def logpdf(self, x: object) -> object: ...
    def marginal(self, dimensions: object) -> object: ...
    __call__ = evaluate
    covariance_factor = scotts_factor
