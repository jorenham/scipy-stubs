from typing import Literal

import numpy as np

def get_poly_vinit(
    kind: Literal["poly", "vinit"],
    dtype: type[np.uint32 | np.uint64],
) -> np.ndarray[tuple[int, int], np.dtype[np.uint32 | np.uint64]]: ...
