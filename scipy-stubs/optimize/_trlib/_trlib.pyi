from typing_extensions import override

import numpy as np
import numpy.typing as npt
from scipy._typing import UntypedCallable
from scipy.optimize._trustregion import BaseQuadraticSubproblem

__all__ = ["TRLIBQuadraticSubproblem"]

class TRLIBQuadraticSubproblem(BaseQuadraticSubproblem):
    def __init__(
        self,
        /,
        x,
        fun: UntypedCallable,
        jac: UntypedCallable,
        hess: UntypedCallable | None,
        hessp: UntypedCallable | None,
        tol_rel_i: float = -2.0,
        tol_rel_b: float = -3.0,
        disp: bool = False,
    ) -> None: ...
    @override
    def solve(self, trust_radius: float | np.float64) -> tuple[npt.NDArray[np.float64], bool]: ...
