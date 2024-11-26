import numpy as np
import optype.numpy as onp

def levinson(
    a: onp.ArrayND[np.float64 | np.complex128],
    b: onp.ArrayND[np.float64 | np.complex128],
) -> tuple[onp.Array1D[np.float64 | np.complex128]]: ...
