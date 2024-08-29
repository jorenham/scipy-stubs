# TODO
from scipy._typing import Untyped

__all__ = ["laplacian"]

def laplacian(
    csgraph: Untyped,
    normed: bool = False,
    return_diag: bool = False,
    use_out_degree: bool = False,
    *,
    copy: bool = True,
    form: str = "array",
    dtype: Untyped | None = None,
    symmetrized: bool = False,
) -> Untyped: ...
