from collections.abc import Mapping
from typing import Final
from typing_extensions import Never, override

import numpy as np
import numpy.typing as npt
from scipy._typing import Untyped, UntypedCallable
from scipy.optimize._trustregion import BaseQuadraticSubproblem

__test__: Final[Mapping[Never, Never]]

class TRLIBQuadraticSubproblem(BaseQuadraticSubproblem):
    def __init__(
        self,
        /,
        x: Untyped,
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
