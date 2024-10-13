from typing import Literal

import numpy.typing as npt
from ._resampling import PermutationMethod, PermutationTestResult

def bws_test(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    *,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    method: PermutationMethod | None = None,
) -> PermutationTestResult: ...
