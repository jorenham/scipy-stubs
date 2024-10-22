from typing import Any, Literal
from typing_extensions import Self

import numpy as np
import numpy.typing as npt
from numpy._typing import _ArrayLikeFloat_co
from ._crosstab import crosstab
from ._odds_ratio import odds_ratio
from ._relative_risk import relative_risk
from ._typing import BaseBunch, PowerDivergenceStatistic

__all__ = ["association", "chi2_contingency", "crosstab", "expected_freq", "margins", "odds_ratio", "relative_risk"]

class Chi2ContingencyResult(BaseBunch[np.float64, np.float64, int, npt.NDArray[np.float64]]):
    @property
    def statistic(self, /) -> np.float64: ...
    @property
    def pvalue(self, /) -> np.float64: ...
    @property
    def dof(self, /) -> int: ...
    @property
    def expected_freq(self, /) -> npt.NDArray[np.float64]: ...
    def __new__(
        _cls,
        statistic: np.float64,
        pvalue: np.float64,
        dof: int,
        expected_freq: npt.NDArray[np.float64],
    ) -> Self: ...
    def __init__(
        self,
        /,
        statistic: np.float64,
        pvalue: np.float64,
        dof: int,
        expected_freq: npt.NDArray[np.float64],
    ) -> None: ...

def margins(a: npt.NDArray[np.number[Any] | np.bool_ | np.timedelta64]) -> list[npt.NDArray[np.number[Any] | np.timedelta64]]: ...
def expected_freq(observed: _ArrayLikeFloat_co) -> np.float64 | npt.NDArray[np.float64]: ...
def chi2_contingency(
    observed: _ArrayLikeFloat_co,
    correction: bool = True,
    lambda_: PowerDivergenceStatistic | float | None = None,
) -> Chi2ContingencyResult: ...
def association(
    observed: _ArrayLikeFloat_co,
    method: Literal["cramer", "tschuprow", "pearson"] = "cramer",
    correction: bool = False,
    lambda_: PowerDivergenceStatistic | float | None = None,
) -> float: ...
