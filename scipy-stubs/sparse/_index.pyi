from typing import Any

import numpy as np
import optype.numpy as onpt
from scipy._typing import Untyped

INT_TYPES: tuple[type[int], type[np.integer[Any]]] = ...

class IndexMixin:
    def __getitem__(self, key: onpt.AnyIntegerArray, /) -> Untyped: ...
    def __setitem__(self, key: onpt.AnyIntegerArray, x: Untyped, /) -> None: ...
