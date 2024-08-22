from scipy._typing import Untyped

BSR_ROUTINES: str
CSC_ROUTINES: str
CSR_ROUTINES: str
OTHER_ROUTINES: str
COMPILATION_UNITS: Untyped
I_TYPES: Untyped
T_TYPES: Untyped
THUNK_TEMPLATE: str
METHOD_TEMPLATE: str
GET_THUNK_CASE_TEMPLATE: str

def newer(source, target) -> Untyped: ...
def get_thunk_type_set() -> Untyped: ...
def parse_routine(name, args, types) -> Untyped: ...
def main(): ...
def write_autogen_blurb(stream): ...
