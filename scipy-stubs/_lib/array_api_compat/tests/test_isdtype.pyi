from ._helpers import import_ as import_, wrapped_libraries as wrapped_libraries
from scipy._typing import Untyped

dtype_categories: Untyped

def isdtype_(dtype_, kind) -> Untyped: ...
def test_isdtype_spec_dtypes(library): ...

additional_dtypes: Untyped

def test_isdtype_additional_dtypes(library, dtype_): ...
