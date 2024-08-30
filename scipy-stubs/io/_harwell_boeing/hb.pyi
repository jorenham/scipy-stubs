from scipy._typing import Untyped
from scipy.sparse import csc_matrix as csc_matrix
from ._fortran_format_parser import ExpFormat as ExpFormat, FortranFormatParser as FortranFormatParser, IntFormat as IntFormat

class MalformedHeader(Exception): ...
class LineOverflow(Warning): ...

class HBInfo:
    @classmethod
    def from_data(
        cls, m, title: str = "Default title", key: str = "0", mxtype: Untyped | None = None, fmt: Untyped | None = None
    ) -> Untyped: ...
    @classmethod
    def from_file(cls, fid) -> Untyped: ...
    title: Untyped
    key: Untyped
    total_nlines: Untyped
    pointer_nlines: Untyped
    indices_nlines: Untyped
    values_nlines: Untyped
    pointer_format: Untyped
    indices_format: Untyped
    values_format: Untyped
    pointer_dtype: Untyped
    indices_dtype: Untyped
    values_dtype: Untyped
    pointer_nbytes_full: Untyped
    indices_nbytes_full: Untyped
    values_nbytes_full: Untyped
    nrows: Untyped
    ncols: Untyped
    nnon_zeros: Untyped
    nelementals: Untyped
    mxtype: Untyped
    def __init__(
        self,
        title,
        key,
        total_nlines,
        pointer_nlines,
        indices_nlines,
        values_nlines,
        mxtype,
        nrows,
        ncols,
        nnon_zeros,
        pointer_format_str,
        indices_format_str,
        values_format_str,
        right_hand_sides_nlines: int = 0,
        nelementals: int = 0,
    ): ...
    def dump(self) -> Untyped: ...

class HBMatrixType:
    @classmethod
    def from_fortran(cls, fmt) -> Untyped: ...
    value_type: Untyped
    structure: Untyped
    storage: Untyped
    def __init__(self, value_type, structure, storage: str = "assembled"): ...
    @property
    def fortran_format(self) -> Untyped: ...

class HBFile:
    def __init__(self, file, hb_info: Untyped | None = None): ...
    @property
    def title(self) -> Untyped: ...
    @property
    def key(self) -> Untyped: ...
    @property
    def type(self) -> Untyped: ...
    @property
    def structure(self) -> Untyped: ...
    @property
    def storage(self) -> Untyped: ...
    def read_matrix(self) -> Untyped: ...
    def write_matrix(self, m) -> Untyped: ...

def hb_read(path_or_open_file) -> Untyped: ...
def hb_write(path_or_open_file, m, hb_info: Untyped | None = None) -> Untyped: ...
