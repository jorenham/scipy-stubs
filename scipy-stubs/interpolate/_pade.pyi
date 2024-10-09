import numpy as np
import numpy.typing as npt

__all__ = ["pade"]

def pade(an: npt.ArrayLike, m: int, n: int | None = None) -> tuple[np.poly1d, np.poly1d]: ...
