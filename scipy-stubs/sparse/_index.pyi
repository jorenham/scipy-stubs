from typing import Any

import numpy as np
import optype.numpy as onp
from scipy._typing import Untyped

INT_TYPES: tuple[type[int], type[np.integer[Any]]] = ...

class IndexMixin:
    def __getitem__(self, key: onp.AnyIntegerArray, /) -> Untyped: ...
    def __setitem__(self, key: onp.AnyIntegerArray, x: Untyped, /) -> None: ...
