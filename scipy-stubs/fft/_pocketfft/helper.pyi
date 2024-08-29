from collections.abc import Generator

from scipy._typing import Untyped

from scipy._lib._util import copy_if_needed as copy_if_needed
from .pypocketfft import good_size as good_size, prev_good_size as prev_good_size

def set_workers(workers) -> Generator[None, None, None]: ...
def get_workers() -> Untyped: ...
