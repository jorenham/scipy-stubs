from scipy._typing import Untyped

class ReportBase:
    COLUMN_NAMES: list[str]
    COLUMN_WIDTHS: list[int]
    ITERATION_FORMATS: list[str]
    @classmethod
    def print_header(cls): ...
    @classmethod
    def print_iteration(cls, *args): ...
    @classmethod
    def print_footer(cls): ...

class BasicReport(ReportBase):
    COLUMN_NAMES: Untyped
    COLUMN_WIDTHS: Untyped
    ITERATION_FORMATS: Untyped

class SQPReport(ReportBase):
    COLUMN_NAMES: Untyped
    COLUMN_WIDTHS: Untyped
    ITERATION_FORMATS: Untyped

class IPReport(ReportBase):
    COLUMN_NAMES: Untyped
    COLUMN_WIDTHS: Untyped
    ITERATION_FORMATS: Untyped
