import numpy as np
import optype.numpy as onpt
from optype import CanBool, CanFloat, CanInt

def _cy_wrapper_centered_discrepancy(
    sample: onpt.Array[tuple[int, int], np.float64],
    iterative: CanBool,
    workers: CanInt,
) -> float: ...
def _cy_wrapper_wrap_around_discrepancy(
    sample: onpt.Array[tuple[int, int], np.float64],
    iterative: CanBool,
    workers: CanInt,
) -> float: ...
def _cy_wrapper_mixture_discrepancy(
    sample: onpt.Array[tuple[int, int], np.float64],
    iterative: CanBool,
    workers: CanInt,
) -> float: ...
def _cy_wrapper_l2_star_discrepancy(
    sample: onpt.Array[tuple[int, int], np.float64],
    iterative: CanBool,
    workers: CanInt,
) -> float: ...
def _cy_wrapper_update_discrepancy(
    x_new_view: onpt.Array[tuple[int], np.float64],
    sample_view: onpt.Array[tuple[int, int], np.float64],
    initial_disc: CanFloat,
) -> float: ...
def _cy_van_der_corput(
    n: CanInt,
    base: CanInt,
    start_index: CanInt,
    workers: CanInt,
) -> onpt.Array[tuple[int], np.float64]: ...
def _cy_van_der_corput_scrambled(
    n: CanInt,
    base: CanInt,
    start_index: CanInt,
    permutations: onpt.Array[tuple[int, int], np.int64],
    workers: CanInt,
) -> onpt.Array[tuple[int], np.float64]: ...
