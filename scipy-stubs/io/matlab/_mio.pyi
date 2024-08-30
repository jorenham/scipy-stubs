from scipy._typing import Untyped
from ._mio4 import MatFile4Reader as MatFile4Reader, MatFile4Writer as MatFile4Writer
from ._mio5 import MatFile5Reader as MatFile5Reader, MatFile5Writer as MatFile5Writer
from ._miobase import docfiller as docfiller

def mat_reader_factory(file_name, appendmat: bool = True, **kwargs) -> Untyped: ...
def loadmat(file_name, mdict: Untyped | None = None, appendmat: bool = True, **kwargs) -> Untyped: ...
def savemat(
    file_name,
    mdict,
    appendmat: bool = True,
    format: str = "5",
    long_field_names: bool = False,
    do_compression: bool = False,
    oned_as: str = "row",
): ...
def whosmat(file_name, appendmat: bool = True, **kwargs) -> Untyped: ...
