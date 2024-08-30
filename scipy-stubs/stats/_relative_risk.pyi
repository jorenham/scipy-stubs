from dataclasses import dataclass

from scipy._typing import Untyped
from scipy.special import ndtri as ndtri
from ._common import ConfidenceInterval as ConfidenceInterval

@dataclass
class RelativeRiskResult:
    relative_risk: float
    exposed_cases: int
    exposed_total: int
    control_cases: int
    control_total: int
    def confidence_interval(self, confidence_level: float = 0.95) -> Untyped: ...

def relative_risk(exposed_cases, exposed_total, control_cases, control_total) -> Untyped: ...
