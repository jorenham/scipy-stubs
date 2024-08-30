from typing import Any
from typing_extensions import override

import numpy as np
import numpy.typing as npt
import optype.numpy as onpt

class NDInterpolatorBase:
    def __init__(
        self,
        /,
        points: onpt.AnyIntegerArray | onpt.AnyFloatingArray,
        values: onpt.AnyFloatingArray,
        fill_value: float = ...,
        ndim: int | None = None,
        rescale: bool = False,
        need_contiguous: bool = True,
        need_values: bool = True,
    ) -> None: ...
    def __call__(self, /, *args: onpt.AnyFloatingArray) -> npt.NDArray[np.floating[Any]]: ...

class LinearNDInterpolator(NDInterpolatorBase):
    @override
    def __init__(
        self,
        /,
        points: onpt.AnyIntegerArray | onpt.AnyFloatingArray,
        values: onpt.AnyFloatingArray,
        fill_value: float = ...,
        rescale: bool = False,
    ) -> None: ...

class CloughTocher2DInterpolator(NDInterpolatorBase):
    @override
    def __init__(
        self,
        /,
        points: onpt.AnyIntegerArray | onpt.AnyFloatingArray,
        values: onpt.AnyFloatingArray,
        fill_value: float = ...,
        tol: float = 1e-06,
        maxiter: int = 400,
        rescale: bool = False,
    ) -> None: ...
