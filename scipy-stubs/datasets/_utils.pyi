from ._typing import Fetcher

# NOTE: the implementation explcitily checks for `list` and `tuple` types
def clear_cache(datasets: Fetcher | list[Fetcher] | tuple[Fetcher, ...] | None = None) -> None: ...
