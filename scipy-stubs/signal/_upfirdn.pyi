from typing import Any, Literal, TypeAlias

import numpy as np
import numpy.typing as npt
import optype.numpy as onpt

__all__ = ["_output_len", "upfirdn"]

_FIRMode: TypeAlias = Literal["constant", "symmetric", "reflect", "wrap"]
_int64_t: TypeAlias = int | np.int64  # noqa: PYI042

class _UpFIRDn:
    def __init__(self, /, h: npt.NDArray[np.floating[Any]], x_dtype: np.dtype[np.floating[Any]], up: int, down: int) -> None: ...
    def apply_filter(
        self,
        x: npt.NDArray[np.number[Any]],
        axis: int = -1,
        mode: _FIRMode = "constant",
        cval: int = 0,
    ) -> npt.NDArray[np.floating[Any]]: ...

def upfirdn(
    h: onpt.AnyFloatingArray,
    x: onpt.AnyIntegerArray | onpt.AnyFloatingArray,
    up: int = 1,
    down: int = 1,
    axis: int = -1,
    mode: _FIRMode = "constant",
    cval: float = 0,
) -> npt.NDArray[np.floating[Any]]: ...

# originally defined in `scipy/signal/_upfirdn_apply.pyx` (as `(((in_len - 1) * up + len_h) - 1) // down + 1`)
def _output_len(len_h: _int64_t, in_len: _int64_t, up: _int64_t, down: _int64_t) -> _int64_t: ...
