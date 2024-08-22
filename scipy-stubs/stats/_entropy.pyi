import numpy as np

from scipy import special as special
from scipy._lib._array_api import array_namespace as array_namespace, xp_moveaxis_to_end as xp_moveaxis_to_end

def entropy(
    pk: np.typing.ArrayLike, qk: np.typing.ArrayLike | None = None, base: float | None = None, axis: int = 0
) -> np.number | np.ndarray: ...
def differential_entropy(
    values: np.typing.ArrayLike,
    *,
    window_length: int | None = None,
    base: float | None = None,
    axis: int = 0,
    method: str = "auto",
) -> np.number | np.ndarray: ...
