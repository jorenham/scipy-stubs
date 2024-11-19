from dataclasses import dataclass
from typing import Literal

import numpy as np
import optype as op
from numpy._typing import _ArrayLikeFloat_co, _ArrayLikeInt

@dataclass
class PageTrendTestResult:
    statistic: np.float64
    pvalue: np.float64
    method: Literal["asymptotic", "exact"]

def page_trend_test(
    data: _ArrayLikeFloat_co,
    ranked: op.CanBool = False,
    predicted_ranks: _ArrayLikeInt | None = None,
    method: Literal["auto", "asymptotic", "exact"] = "auto",
) -> PageTrendTestResult: ...
