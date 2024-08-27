from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt

__all__ = ["qr_delete", "qr_insert", "qr_update"]

_Which: TypeAlias = Literal["row", "col"]

def qr_delete(
    Q: npt.ArrayLike,
    R: npt.ArrayLike,
    k: int,
    p: int = 1,
    which: _Which = "row",
    overwrite_qr: bool = False,
    check_finite: bool = True,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
def qr_insert(
    Q: npt.ArrayLike,
    R: npt.ArrayLike,
    u: npt.ArrayLike,
    k: int,
    which: _Which = "row",
    rcond: float | None = None,
    overwrite_qru: bool = False,
    check_finite: bool = True,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
def qr_update(
    Q: npt.ArrayLike,
    R: npt.ArrayLike,
    u: npt.ArrayLike,
    v: npt.ArrayLike,
    overwrite_qruv: bool = False,
    check_finite: bool = True,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
