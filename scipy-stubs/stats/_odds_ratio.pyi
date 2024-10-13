from typing import Any, Literal, TypeAlias

import numpy as np
import optype.numpy as onpt
from numpy._typing import _ArrayLikeInt_co
from ._common import ConfidenceInterval

_Kind: TypeAlias = Literal["conditional", "sample"]

class OddsRatioResult:
    statistic: float
    def __init__(
        self,
        _table: onpt.Array[tuple[Literal[2], Literal[2]], np.integer[Any]],
        _kind: _Kind,
        statistic: float,
    ) -> None: ...
    def confidence_interval(self, confidence_level: float = 0.95, alternative: str = "two-sided") -> ConfidenceInterval: ...

def odds_ratio(table: _ArrayLikeInt_co, *, kind: _Kind = "conditional") -> OddsRatioResult: ...
