from typing import Final
from typing_extensions import LiteralString

from scipy._typing import Untyped

__pythran__: Final[tuple[LiteralString, LiteralString]]

def group_dense(m: Untyped, n: Untyped, A: Untyped) -> Untyped: ...
def group_sparse(m: Untyped, n: Untyped, indices: Untyped, indptr: Untyped) -> Untyped: ...
