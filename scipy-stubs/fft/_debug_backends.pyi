from scipy._typing import Untyped

class NumPyBackend:
    __ua_domain__: str
    @staticmethod
    def __ua_function__(method: Untyped, args: Untyped, kwargs: Untyped) -> Untyped: ...

class EchoBackend:
    __ua_domain__: str
    @staticmethod
    def __ua_function__(method: Untyped, args: Untyped, kwargs: Untyped) -> None: ...
