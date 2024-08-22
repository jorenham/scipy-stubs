from . import distributions as distributions
from scipy._typing import Untyped

TheilslopesResult: Untyped
SiegelslopesResult: Untyped

def theilslopes(y, x: Untyped | None = None, alpha: float = 0.95, method: str = "separate") -> Untyped: ...
def siegelslopes(y, x: Untyped | None = None, method: str = "hierarchical") -> Untyped: ...
