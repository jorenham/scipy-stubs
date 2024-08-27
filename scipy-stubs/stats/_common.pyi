from typing import NamedTuple

__all__ = ["ConfidenceInterval"]

class ConfidenceInterval(NamedTuple):
    low: float
    high: float
