from typing import Final, Literal

import numpy as np
import optype.numpy as onp

N_STAGES: Final = 12
N_STAGES_EXTENDED: Final = 16
INTERPOLATOR_POWER: Final = 7
C: Final[onp.Array[tuple[Literal[16]], np.float64]]
A: Final[onp.Array[tuple[Literal[16], Literal[16]], np.float64]]
B: Final[onp.Array[tuple[Literal[12]], np.float64]]
E3: Final[onp.Array[tuple[Literal[13]], np.float64]]
E5: Final[onp.Array[tuple[Literal[13]], np.float64]]
D: Final[onp.Array[tuple[Literal[4, 16]], np.float64]]
