class DegenerateDataWarning(RuntimeWarning):
    args: tuple[str | None]
    def __init__(self, msg: str | None = None) -> None: ...

class ConstantInputWarning(DegenerateDataWarning): ...
class NearConstantInputWarning(DegenerateDataWarning): ...
class FitError(RuntimeError): ...
