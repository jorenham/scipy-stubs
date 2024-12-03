from typing import TypeAlias

import numpy as np
import optype.numpy as onp

__all__ = ["pade_UV_calc", "pick_pade_structure"]

_Inexact: TypeAlias = np.float32 | np.float64 | np.complex64 | np.complex128

# `Am` must have a shape like `(5, n, n)`.
def pick_pade_structure(Am: onp.ArrayND[_Inexact]) -> tuple[int, int]: ...
def pade_UV_calc(Am: onp.ArrayND[_Inexact], n: int, m: int) -> None: ...
