from typing import final

from ._backend import _BaseBackend

@final
class NumPyBackend(_BaseBackend): ...

@final
class EchoBackend(_BaseBackend[None]): ...
