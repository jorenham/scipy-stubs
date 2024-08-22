from scipy._typing import Untyped

def import_wrappers_common() -> Untyped: ...

C_PREAMBLE: Untyped
C_TYPES: Untyped
CPP_GUARD_BEGIN: Untyped
CPP_GUARD_END: Untyped
LAPACK_DECLS: Untyped
NPY_TYPES: Untyped
WRAPPED_FUNCS: Untyped
all_newer: Untyped
get_blas_macro_and_name: Untyped
read_signatures: Untyped
write_files: Untyped
BASE_DIR: Untyped
COMMENT_TEXT: Untyped
blas_pyx_preamble: str
lapack_pyx_preamble: str
blas_py_wrappers: str
lapack_py_wrappers: str

def arg_casts(argtype) -> Untyped: ...
def generate_decl_pyx(name, return_type, argnames, argtypes, accelerate, header_name) -> Untyped: ...
def generate_file_pyx(sigs, lib_name, header_name, accelerate) -> Untyped: ...

blas_pxd_preamble: str
lapack_pxd_preamble: str

def generate_decl_pxd(name, return_type, argnames, argtypes) -> Untyped: ...
def generate_file_pxd(sigs, lib_name) -> Untyped: ...
def generate_decl_c(name, return_type, argnames, argtypes, accelerate) -> Untyped: ...
def generate_file_c(sigs, lib_name, accelerate) -> Untyped: ...
def make_all(
    outdir,
    blas_signature_file: str = "cython_blas_signatures.txt",
    lapack_signature_file: str = "cython_lapack_signatures.txt",
    blas_name: str = "cython_blas",
    lapack_name: str = "cython_lapack",
    blas_header_name: str = "_blas_subroutines.h",
    lapack_header_name: str = "_lapack_subroutines.h",
    accelerate: bool = False,
): ...
