from collections.abc import Callable
from typing import Literal, TypeAlias

import numpy as np
import optype.numpy as onpt
from scipy._typing import AnyInt, AnyReal, Untyped, UntypedCallable

__all__ = ["fmin_cobyla"]

_Array_1d_f8: TypeAlias = onpt.Array[tuple[int], np.float64]

###

def fmin_cobyla(
    func: UntypedCallable,
    x0: Untyped,
    cons: Untyped,
    args: tuple[object, ...] = (),
    consargs: tuple[object, ...] | None = None,
    rhobeg: AnyReal = 1.0,
    rhoend: AnyReal = 0.0001,
    maxfun: AnyInt = 1000,
    disp: Literal[0, 1, 2, 3] | None = None,
    catol: AnyReal = 0.0002,
    *,
    callback: Callable[[_Array_1d_f8], None] | None = None,
) -> _Array_1d_f8: ...
