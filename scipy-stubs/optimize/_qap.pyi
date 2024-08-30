from scipy._lib._util import check_random_state as check_random_state
from scipy._typing import Untyped
from . import OptimizeResult as OptimizeResult, linear_sum_assignment as linear_sum_assignment

QUADRATIC_ASSIGNMENT_METHODS: Untyped

def quadratic_assignment(A, B, method: str = "faq", options: Untyped | None = None) -> Untyped: ...
