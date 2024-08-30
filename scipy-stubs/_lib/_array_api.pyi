from types import ModuleType
from typing import Any, Literal

from scipy._lib.array_api_compat import device, size
from scipy._typing import Untyped

__all__ = ["_asarray", "array_namespace", "device", "size"]

def _asarray(
    array: Untyped,
    dtype: Any = None,
    order: Literal["K", "A", "C", "F"] | None = None,
    copy: bool | None = None,
    *,
    xp: ModuleType | None = None,
    check_finite: bool = False,
    subok: bool = False,
) -> Untyped: ...
def array_namespace(*arrays: Untyped) -> ModuleType: ...
