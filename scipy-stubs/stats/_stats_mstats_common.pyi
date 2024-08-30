from scipy._typing import Untyped
from . import distributions as distributions

TheilslopesResult: Untyped
SiegelslopesResult: Untyped

def theilslopes(y, x: Untyped | None = None, alpha: float = 0.95, method: str = "separate") -> Untyped: ...
def siegelslopes(y, x: Untyped | None = None, method: str = "hierarchical") -> Untyped: ...
