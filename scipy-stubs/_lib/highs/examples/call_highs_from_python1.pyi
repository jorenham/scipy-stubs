import typing

from scipy._typing import Untyped

inf: Untyped
h: Untyped
alt_inf: Untyped
num_cons: int
lower: Untyped
upper: Untyped
num_new_nz: int
starts: Untyped
indices: Untyped
values: Untyped
num_var: Untyped
solution: Untyped
basis: Untyped
info: Untyped
model_status: Untyped
lp: Untyped
num_nz: Untyped
num_row: Untyped
model: Untyped
num_col: int
sense: Untyped
offset: int
col_cost: Untyped
col_lower: Untyped
col_upper: Untyped
row_lower: Untyped
row_upper: Untyped
a_matrix_format: Untyped
a_matrix_start: Untyped
a_matrix_index: Untyped
a_matrix_value: Untyped
a_matrix_num_nz: typing.TypeAlias = a_matrix_start[num_col]
hessian_format: Untyped
hessian_start: Untyped
hessian_index: Untyped
hessian_value: Untyped
hessian_num_nz: typing.TypeAlias = hessian_start[num_col]
integrality: Untyped
presolved_lp: Untyped
h1: Untyped
options: Untyped
run_time: Untyped
