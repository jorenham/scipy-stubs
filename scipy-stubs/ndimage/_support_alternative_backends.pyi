from ._ndimage_api import *
from scipy._lib._array_api import (
    SCIPY_ARRAY_API as SCIPY_ARRAY_API,
    is_cupy as is_cupy,
    is_jax as is_jax,
    scipy_namespace_for as scipy_namespace_for,
)
from scipy._typing import Untyped

MODULE_NAME: str

def dispatch_xp(dispatcher, module_name) -> Untyped: ...

bare_func: Untyped
dispatcher: Untyped
f: Untyped
