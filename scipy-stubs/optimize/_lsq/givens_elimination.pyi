import numpy as np
import optype.numpy as onpt

def givens_elimination(
    S: onpt.Array[tuple[int, int], np.float64],
    v: onpt.Array[tuple[int], np.float64],
    diag: onpt.Array[tuple[int], np.float64],
) -> None: ...
