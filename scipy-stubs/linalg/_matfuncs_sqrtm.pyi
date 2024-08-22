import numpy as np

from ._decomp_schur import rsf2csf as rsf2csf, schur as schur
from ._matfuncs_sqrtm_triu import within_block_loop as within_block_loop
from ._misc import norm as norm
from .lapack import dtrsyl as dtrsyl, ztrsyl as ztrsyl
from scipy._typing import Untyped

class SqrtmError(np.linalg.LinAlgError): ...

def sqrtm(A, disp: bool = True, blocksize: int = 64) -> Untyped: ...
