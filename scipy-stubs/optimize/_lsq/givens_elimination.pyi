from collections.abc import Mapping
from typing import Final
from typing_extensions import Never

import numpy as np
import optype.numpy as onpt

__test__: Final[Mapping[Never, Never]]

def givens_elimination(
    S: onpt.Array[tuple[int, int], np.float64],
    v: onpt.Array[tuple[int], np.float64],
    diag: onpt.Array[tuple[int], np.float64],
) -> None: ...
