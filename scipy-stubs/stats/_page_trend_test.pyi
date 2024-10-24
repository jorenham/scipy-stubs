from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy._typing import _ArrayLikeFloat_co, _ArrayLikeInt
from optype import CanBool

@dataclass
class PageTrendTestResult:
    statistic: np.float64
    pvalue: np.float64
    method: Literal["asymptotic", "exact"]

def page_trend_test(
    data: _ArrayLikeFloat_co,
    ranked: CanBool = False,
    predicted_ranks: _ArrayLikeInt | None = None,
    method: Literal["auto", "asymptotic", "exact"] = "auto",
) -> PageTrendTestResult: ...
