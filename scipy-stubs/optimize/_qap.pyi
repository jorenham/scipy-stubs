from typing import Final

from scipy._typing import Untyped

QUADRATIC_ASSIGNMENT_METHODS: Final = ["faq", "2opt"]

def quadratic_assignment(A: Untyped, B: Untyped, method: str = "faq", options: Untyped | None = None) -> Untyped: ...
