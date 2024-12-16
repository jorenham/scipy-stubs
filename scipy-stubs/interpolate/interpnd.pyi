from typing import Any
from typing_extensions import deprecated

from . import _interpnd

__all__ = [
    "CloughTocher2DInterpolator",
    "GradientEstimationWarning",
    "LinearNDInterpolator",
    "NDInterpolatorBase",
    "estimate_gradients_2d_global",
]

# NOTE: See https://github.com/scipy/scipy/issues/22097 for `GradientEstimationWarning` and `NDInterpolatorBase`

@deprecated(
    "`scipy.interpolate.interpnd.GradientEstimationWarning` is deprecated along with the `scipy.interpolate.interpnd` namespace. "
    "`scipy.interpolate.interpnd.GradientEstimationWarning` will be removed in SciPy 1.14.0, "
    "and the `scipy.interpolate.interpnd` namespace will be removed in SciPy 2.0.0."
)
class GradientEstimationWarning(_interpnd.GradientEstimationWarning): ...

@deprecated(
    "`scipy.interpolate.interpnd.NDInterpolatorBase` is deprecated along with the `scipy.interpolate.interpnd` namespace. "
    "`scipy.interpolate.interpnd.NDInterpolatorBase` will be removed in SciPy 1.14.0, and the `scipy.interpolate.interpnd` "
    "namespace will be removed in SciPy 2.0.0."
)
class NDInterpolatorBase(_interpnd.NDInterpolatorBase): ...

@deprecated(
    "Please import `CloughTocher2DInterpolator` from the `scipy.interpolate` namespace; "
    "the `scipy.interpolate.interpnd` namespace is deprecated and will be removed in SciPy 2.0.0."
)
class CloughTocher2DInterpolator(_interpnd.CloughTocher2DInterpolator): ...

@deprecated(
    "Please import `LinearNDInterpolator` from the `scipy.interpolate` namespace; "
    "the `scipy.interpolate.interpnd` namespace is deprecated and will be removed in SciPy 2.0.0."
)
class LinearNDInterpolator(_interpnd.LinearNDInterpolator): ...

@deprecated(
    "`scipy.interpolate.interpnd.estimate_gradients_2d_global` is deprecated along with the `scipy.interpolate.interpnd` "
    "namespace. `scipy.interpolate.interpnd.estimate_gradients_2d_global` will be removed in SciPy 1.14.0, and the "
    "`scipy.interpolate.interpnd` namespace will be removed in SciPy 2.0.0."
)
def estimate_gradients_2d_global(tri: object, y: object, maxiter: int = 400, tol: float = 1e-6) -> Any: ...  # noqa: ANN401
