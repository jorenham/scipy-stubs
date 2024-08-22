from ctypes.util import find_library as find_library

from scipy._typing import Untyped

highslib: Untyped

def Highs_lpCall(col_cost, col_lower, col_upper, row_lower, row_upper, a_start, a_index, a_value) -> Untyped: ...
def Highs_mipCall(col_cost, col_lower, col_upper, row_lower, row_upper, a_start, a_index, a_value, integrality) -> Untyped: ...
def Highs_qpCall(
    col_cost, col_lower, col_upper, row_lower, row_upper, a_start, a_index, a_value, q_start, q_index, q_value
) -> Untyped: ...
