from collections.abc import Iterable, Sequence
from typing import Final, Literal

import numpy.typing as npt
import optype.numpy as onp
import scipy._typing as spt

__all__ = ["get_lapack_funcs"]

HAS_ILP64: Final[bool]

def get_lapack_funcs(
    names: Iterable[str] | str,
    arrays: Sequence[onp.ArrayND] = (),
    dtype: npt.DTypeLike | None = None,
    ilp64: Literal["preferred"] | bool = False,
) -> list[spt._FortranFunction] | spt._FortranFunction: ...
