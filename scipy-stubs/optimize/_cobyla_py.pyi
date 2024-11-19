from collections.abc import Callable
from typing import Literal

import numpy as np
import optype.numpy as onp
from scipy._typing import Untyped, UntypedCallable

__all__ = ["fmin_cobyla"]

###

def fmin_cobyla(
    func: UntypedCallable,
    x0: Untyped,
    cons: Untyped,
    args: tuple[object, ...] = (),
    consargs: tuple[object, ...] | None = None,
    rhobeg: onp.ToFloat = 1.0,
    rhoend: onp.ToFloat = 0.0001,
    maxfun: onp.ToInt = 1000,
    disp: Literal[0, 1, 2, 3] | None = None,
    catol: onp.ToFloat = 0.0002,
    *,
    callback: Callable[[onp.Array1D[np.float64]], None] | None = None,
) -> onp.Array1D[np.float64]: ...
