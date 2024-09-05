from collections.abc import Callable
from typing import TypeAlias
from typing_extensions import TypeVar

import numpy as np

_ShapeT = TypeVar("_ShapeT", bound=tuple[int, ...], default=tuple[int] | tuple[int, int] | tuple[int, int, int])
_DT = TypeVar("_DT", bound=np.dtype[np.generic], default=np.dtype[np.float64] | np.dtype[np.uint8])

_AnyDataset: TypeAlias = Callable[[], np.ndarray[_ShapeT, _DT]]

def clear_cache(datasets: list[_AnyDataset] | tuple[_AnyDataset, ...] | None = None) -> None: ...
