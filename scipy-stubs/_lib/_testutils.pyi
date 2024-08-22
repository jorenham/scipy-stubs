from scipy._typing import Untyped

required_version: str
IS_MUSL: bool
IS_EDITABLE: Untyped

class FPUModeChangeWarning(RuntimeWarning): ...

class PytestTester:
    module_name: Untyped
    def __init__(self, module_name) -> None: ...
    def __call__(
        self,
        label: str = "fast",
        verbose: int = 1,
        extra_argv: Untyped | None = None,
        doctests: bool = False,
        coverage: bool = False,
        tests: Untyped | None = None,
        parallel: Untyped | None = None,
    ) -> Untyped: ...

class _TestPythranFunc:
    ALL_INTEGER: Untyped
    ALL_FLOAT: Untyped
    ALL_COMPLEX: Untyped
    arguments: Untyped
    partialfunc: Untyped
    expected: Untyped
    def setup_method(self): ...
    def get_optional_args(self, func) -> Untyped: ...
    def get_max_dtype_list_length(self) -> Untyped: ...
    def get_dtype(self, dtype_list, dtype_idx) -> Untyped: ...
    def test_all_dtypes(self): ...
    def test_views(self): ...
    def test_strided(self): ...

def check_free_memory(free_mb): ...
