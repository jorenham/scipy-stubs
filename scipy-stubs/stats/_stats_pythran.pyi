from typing import Final, Literal, TypeAlias

from numpy._typing import _ArrayLikeFloat_co
from ._stats_mstats_common import SiegelslopesResult

_Method: TypeAlias = Literal["hierarchical", "separate"]

###

__pythran__: Final[tuple[str, str]] = ...

def siegelslopes(
    y: _ArrayLikeFloat_co,
    x: _ArrayLikeFloat_co | None = None,
    method: _Method = "hierarchical",
) -> SiegelslopesResult: ...
