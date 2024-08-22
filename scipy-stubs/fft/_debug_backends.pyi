from scipy._typing import Untyped

class NumPyBackend:
    __ua_domain__: str
    @staticmethod
    def __ua_function__(method, args, kwargs) -> Untyped: ...

class EchoBackend:
    __ua_domain__: str
    @staticmethod
    def __ua_function__(method, args, kwargs): ...
