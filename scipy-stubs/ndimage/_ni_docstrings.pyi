__all__ = ["docfiller"]

from typing import Final, TypeVar

import optype as op

_F = TypeVar("_F", bound=op.HasDoc)

#
docdict: Final[dict[str, str]]  # undocumented

#
def docfiller(f: _F) -> _F: ...  # undocumented
