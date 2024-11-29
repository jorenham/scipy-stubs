from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Literal, TypeAlias, TypeVar

import numpy as np
import optype.numpy as onp
from scipy._typing import Untyped, UntypedCallable
from ._optimize import OptimizeResult as _OptimizeResult
from ._typing import Constraints

__all__ = ["shgo"]

_Float: TypeAlias = float | np.float64
_Float1D: TypeAlias = onp.Array1D[np.float64]

_MinimizerKwargs: TypeAlias = Mapping[str, object]  # TODO(jorenham): TypedDict
_Options: TypeAlias = Mapping[str, object]  # TODO(jorenham): TypedDict

_SamplingMethodName: TypeAlias = Literal["simplicial", "halton", "sobol"]
_SamplingMethodFunc: TypeAlias = Callable[[int, int], onp.ArrayND[np.float64]]
_SamplingMethod: TypeAlias = _SamplingMethodName | _SamplingMethodFunc

_VT = TypeVar("_VT")
_RT = TypeVar("_RT")

###

class OptimizeResult(_OptimizeResult):
    x: _Float1D
    xl: list[_Float1D]
    fun: _Float
    funl: list[_Float]
    success: bool
    message: str
    nfev: int
    nlfev: int
    nljev: int  # undocumented
    nlhev: int  # undocumented
    nit: int

def shgo(
    func: UntypedCallable,
    bounds: Untyped,
    args: tuple[object, ...] = (),
    constraints: Constraints | None = None,
    n: int = 100,
    iters: int = 1,
    callback: Callable[[_Float1D], None] | None = None,
    minimizer_kwargs: _MinimizerKwargs | None = None,  # TODO(jorenham): TypedDict
    options: _Options | None = None,  # TODO(jorenham): TypedDict
    sampling_method: _SamplingMethod = "simplicial",
    *,
    workers: int | Callable[[Callable[[_VT], _RT], Iterable[_VT]], Sequence[_RT]] = 1,
) -> OptimizeResult: ...
