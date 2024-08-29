from scipy._typing import Untyped

__all__ = [
    "block_array",
    "block_diag",
    "bmat",
    "diags",
    "diags_array",
    "eye",
    "eye_array",
    "hstack",
    "identity",
    "kron",
    "kronsum",
    "rand",
    "random",
    "random_array",
    "spdiags",
    "vstack",
]

def spdiags(
    data: Untyped,
    diags: Untyped,
    m: Untyped | None = None,
    n: Untyped | None = None,
    format: Untyped | None = None,
) -> Untyped: ...
def diags_array(
    diagonals: Untyped,
    /,
    *,
    offsets: int = 0,
    shape: Untyped | None = None,
    format: Untyped | None = None,
    dtype: Untyped | None = None,
) -> Untyped: ...
def diags(
    diagonals: Untyped,
    offsets: int = 0,
    shape: Untyped | None = None,
    format: Untyped | None = None,
    dtype: Untyped | None = None,
) -> Untyped: ...
def identity(n: Untyped, dtype: str = "d", format: Untyped | None = None) -> Untyped: ...
def eye_array(
    m: Untyped,
    n: Untyped | None = None,
    *,
    k: int = 0,
    dtype: Untyped = ...,
    format: Untyped | None = None,
) -> Untyped: ...
def eye(m: Untyped, n: Untyped | None = None, k: int = 0, dtype: Untyped = ..., format: Untyped | None = None) -> Untyped: ...
def kron(A: Untyped, B: Untyped, format: Untyped | None = None) -> Untyped: ...
def kronsum(A: Untyped, B: Untyped, format: Untyped | None = None) -> Untyped: ...
def hstack(blocks: Untyped, format: Untyped | None = None, dtype: Untyped | None = None) -> Untyped: ...
def vstack(blocks: Untyped, format: Untyped | None = None, dtype: Untyped | None = None) -> Untyped: ...
def bmat(blocks: Untyped, format: Untyped | None = None, dtype: Untyped | None = None) -> Untyped: ...
def block_array(blocks: Untyped, *, format: Untyped | None = None, dtype: Untyped | None = None) -> Untyped: ...
def block_diag(mats: Untyped, format: Untyped | None = None, dtype: Untyped | None = None) -> Untyped: ...
def random_array(
    shape: Untyped,
    *,
    density: float = 0.01,
    format: str = "coo",
    dtype: Untyped | None = None,
    random_state: Untyped | None = None,
    data_sampler: Untyped | None = None,
) -> Untyped: ...
def random(
    m: Untyped,
    n: Untyped,
    density: float = 0.01,
    format: str = "coo",
    dtype: Untyped | None = None,
    random_state: Untyped | None = None,
    data_rvs: Untyped | None = None,
) -> Untyped: ...
def rand(
    m: Untyped,
    n: Untyped,
    density: float = 0.01,
    format: str = "coo",
    dtype: Untyped | None = None,
    random_state: Untyped | None = None,
) -> Untyped: ...
