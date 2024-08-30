from collections.abc import Iterable
from typing import overload

import numpy as np
import numpy.typing as npt

@overload
def within_block_loop(
    R: npt.NDArray[np.float64],
    T: npt.NDArray[np.float64],
    start_stop_pairs: Iterable[tuple[int, int]],
    nblocks: int | np.intp,
) -> None: ...
@overload
def within_block_loop(
    R: npt.NDArray[np.complex128],
    T: npt.NDArray[np.complex128],
    start_stop_pairs: Iterable[tuple[int, int]],
    nblocks: int | np.intp,
) -> None: ...
