from scipy._typing import Untyped
from ._byteordercodes import native_code as native_code, swapped_code as swapped_code
from ._mio5_params import (
    MDTYPES as MDTYPES,
    NP_TO_MTYPES as NP_TO_MTYPES,
    NP_TO_MXTYPES as NP_TO_MXTYPES,
    MatlabFunction as MatlabFunction,
    MatlabObject as MatlabObject,
    mat_struct as mat_struct,
    mclass_info as mclass_info,
    miCOMPRESSED as miCOMPRESSED,
    miINT8 as miINT8,
    miMATRIX as miMATRIX,
    miUINT32 as miUINT32,
    miUTF8 as miUTF8,
    mxCELL_CLASS as mxCELL_CLASS,
    mxCHAR_CLASS as mxCHAR_CLASS,
    mxDOUBLE_CLASS as mxDOUBLE_CLASS,
    mxOBJECT_CLASS as mxOBJECT_CLASS,
    mxSPARSE_CLASS as mxSPARSE_CLASS,
    mxSTRUCT_CLASS as mxSTRUCT_CLASS,
)
from ._miobase import (
    MatFileReader as MatFileReader,
    MatReadError as MatReadError,
    MatReadWarning as MatReadWarning,
    MatWriteError as MatWriteError,
    arr_dtype_number as arr_dtype_number,
    arr_to_chars as arr_to_chars,
    docfiller as docfiller,
    matdims as matdims,
    read_dtype as read_dtype,
)

class MatFile5Reader(MatFileReader):
    uint16_codec: Untyped
    def __init__(
        self,
        mat_stream,
        byte_order: Untyped | None = None,
        mat_dtype: bool = False,
        squeeze_me: bool = False,
        chars_as_strings: bool = True,
        matlab_compatible: bool = False,
        struct_as_record: bool = True,
        verify_compressed_data_integrity: bool = True,
        uint16_codec: Untyped | None = None,
        simplify_cells: bool = False,
    ): ...
    def guess_byte_order(self) -> Untyped: ...
    def read_file_header(self) -> Untyped: ...
    def initialize_read(self): ...
    def read_var_header(self) -> Untyped: ...
    def read_var_array(self, header, process: bool = True) -> Untyped: ...
    def get_variables(self, variable_names: Untyped | None = None) -> Untyped: ...
    def list_variables(self) -> Untyped: ...

def varmats_from_mat(file_obj) -> Untyped: ...

class EmptyStructMarker: ...

def to_writeable(source) -> Untyped: ...

NDT_FILE_HDR: Untyped
NDT_TAG_FULL: Untyped
NDT_TAG_SMALL: Untyped
NDT_ARRAY_FLAGS: Untyped

class VarWriter5:
    mat_tag: Untyped
    file_stream: Untyped
    unicode_strings: Untyped
    long_field_names: Untyped
    oned_as: Untyped
    def __init__(self, file_writer) -> None: ...
    def write_bytes(self, arr): ...
    def write_string(self, s): ...
    def write_element(self, arr, mdtype: Untyped | None = None): ...
    def write_smalldata_element(self, arr, mdtype, byte_count): ...
    def write_regular_element(self, arr, mdtype, byte_count): ...
    def write_header(self, shape, mclass, is_complex: bool = False, is_logical: bool = False, nzmax: int = 0): ...
    def update_matrix_tag(self, start_pos): ...
    def write_top(self, arr, name, is_global): ...
    def write(self, arr): ...
    def write_numeric(self, arr): ...
    def write_char(self, arr, codec: str = "ascii"): ...
    def write_sparse(self, arr): ...
    def write_cells(self, arr): ...
    def write_empty_struct(self): ...
    def write_struct(self, arr): ...
    def write_object(self, arr): ...

class MatFile5Writer:
    file_stream: Untyped
    do_compression: Untyped
    unicode_strings: Untyped
    global_vars: Untyped
    long_field_names: Untyped
    oned_as: Untyped
    def __init__(
        self,
        file_stream,
        do_compression: bool = False,
        unicode_strings: bool = False,
        global_vars: Untyped | None = None,
        long_field_names: bool = False,
        oned_as: str = "row",
    ): ...
    def write_file_header(self): ...
    def put_variables(self, mdict, write_header: Untyped | None = None): ...
