from collections.abc import Sequence
from typing import ClassVar

class ReportBase:
    COLUMN_NAMES: ClassVar[Sequence[str]] = ...
    COLUMN_WIDTHS: ClassVar[Sequence[int]] = ...
    ITERATION_FORMATS: ClassVar[Sequence[str]] = ...

    @classmethod
    def print_header(cls) -> None: ...
    @classmethod
    def print_iteration(cls, /, *args: object) -> None: ...
    @classmethod
    def print_footer(cls) -> None: ...

class BasicReport(ReportBase): ...
class SQPReport(ReportBase): ...
class IPReport(ReportBase): ...
