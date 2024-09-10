from typing import Final, Literal

from scipy._typing import Untyped, UntypedArray

__all__ = ["BFGS", "SR1", "HessianUpdateStrategy"]

class HessianUpdateStrategy:
    def initialize(self, /, n: int, approx_type: Literal["hess", "inv_hess"]) -> None: ...
    def update(self, /, delta_x: UntypedArray, delta_grad: UntypedArray) -> None: ...
    def dot(self, /, p: Untyped) -> UntypedArray: ...
    def get_matrix(self, /) -> UntypedArray: ...

class FullHessianUpdateStrategy(HessianUpdateStrategy):
    init_scale: Untyped
    first_iteration: Untyped
    approx_type: Untyped
    B: Untyped
    H: Untyped
    n: Untyped
    def __init__(self, /, init_scale: Literal["auto"] | float | UntypedArray = "auto") -> None: ...

class BFGS(FullHessianUpdateStrategy):
    min_curvature: Final[float]
    exception_strategy: Final[Literal["skip_update", "damp_update"]]
    def __init__(
        self,
        /,
        exception_strategy: Literal["skip_update", "damp_update"] = "skip_update",
        min_curvature: float | None = None,
        init_scale: Literal["auto"] | float | UntypedArray = "auto",
    ) -> None: ...

class SR1(FullHessianUpdateStrategy):
    min_denominator: Final[float]
    def __init__(
        self,
        /,
        min_denominator: float = 1e-08,
        init_scale: Literal["auto"] | float | UntypedArray = "auto",
    ) -> None: ...
