import numpy as np
import numpy.typing as npt

__all__ = ["pade_UV_calc", "pick_pade_structure"]

def pick_pade_structure(Am: npt.NDArray[np.generic]) -> tuple[int, int]: ...
def pade_UV_calc(Am: npt.NDArray[np.generic], n: int, m: int) -> None: ...
