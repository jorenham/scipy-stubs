from scipy._lib._ccallback import LowLevelCallable as LowLevelCallable
from scipy._lib._testutils import PytestTester as PytestTester
from scipy._typing import Untyped

msg: str
np_minversion: str
np_maxversion: str
test: Untyped
submodules: Untyped

def __dir__() -> Untyped: ...
def __getattr__(name) -> Untyped: ...
