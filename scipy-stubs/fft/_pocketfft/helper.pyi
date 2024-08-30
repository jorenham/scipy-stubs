from collections.abc import Generator

from .pypocketfft import good_size, prev_good_size

__all__ = ["get_workers", "good_size", "prev_good_size", "set_workers"]

def set_workers(workers: int) -> Generator[None, None, None]: ...
def get_workers() -> int: ...
