# This file is not meant for public use and will be removed in SciPy v2.0.0.
from typing_extensions import Never, deprecated

__all__: list[Never] = []

@deprecated("will be removed in SciPy 2.0.0.")
def __dir__() -> list[Never]: ...
@deprecated("will be removed in SciPy 2.0.0.")
def __getattr__(name: str) -> Never: ...  # pyright: ignore[reportIncompleteStub]
