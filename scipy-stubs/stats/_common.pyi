from typing import NamedTuple

from scipy._typing import Untyped

class ConfidenceInterval(NamedTuple):
    low: Untyped
    high: Untyped
