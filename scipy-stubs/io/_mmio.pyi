from scipy._typing import Untyped
from scipy.sparse import coo_matrix as coo_matrix, issparse as issparse

def asstr(s) -> Untyped: ...
def mminfo(source) -> Untyped: ...
def mmread(source) -> Untyped: ...
def mmwrite(
    target, a, comment: str = "", field: Untyped | None = None, precision: Untyped | None = None, symmetry: Untyped | None = None
): ...

class MMFile:
    @property
    def rows(self) -> Untyped: ...
    @property
    def cols(self) -> Untyped: ...
    @property
    def entries(self) -> Untyped: ...
    @property
    def format(self) -> Untyped: ...
    @property
    def field(self) -> Untyped: ...
    @property
    def symmetry(self) -> Untyped: ...
    @property
    def has_symmetry(self) -> Untyped: ...
    FORMAT_COORDINATE: str
    FORMAT_ARRAY: str
    FORMAT_VALUES: Untyped
    FIELD_INTEGER: str
    FIELD_UNSIGNED: str
    FIELD_REAL: str
    FIELD_COMPLEX: str
    FIELD_PATTERN: str
    FIELD_VALUES: Untyped
    SYMMETRY_GENERAL: str
    SYMMETRY_SYMMETRIC: str
    SYMMETRY_SKEW_SYMMETRIC: str
    SYMMETRY_HERMITIAN: str
    SYMMETRY_VALUES: Untyped
    DTYPES_BY_FIELD: Untyped
    @staticmethod
    def reader(): ...
    @staticmethod
    def writer(): ...
    @classmethod
    def info(cls, /, source) -> Untyped: ...
    def __init__(self, **kwargs) -> None: ...
    def read(self, source) -> Untyped: ...
    def write(
        self,
        target,
        a,
        comment: str = "",
        field: Untyped | None = None,
        precision: Untyped | None = None,
        symmetry: Untyped | None = None,
    ): ...
