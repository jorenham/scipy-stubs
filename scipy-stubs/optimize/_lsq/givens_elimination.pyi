import numpy as np
import optype.numpy as onp

def givens_elimination(
    S: onp.Array[tuple[int, int], np.float64],
    v: onp.Array[tuple[int], np.float64],
    diag: onp.Array[tuple[int], np.float64],
) -> None: ...
