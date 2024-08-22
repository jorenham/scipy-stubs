from scipy._typing import Untyped

DATA_DIR: str
BOOST_SRC: str
CXX_COMMENT: Untyped
DATA_REGEX: Untyped
ITEM_REGEX: Untyped
HEADER_REGEX: Untyped
IGNORE_PATTERNS: Untyped

def parse_ipp_file(filename) -> Untyped: ...
def dump_dataset(filename, data): ...
def dump_datasets(filename): ...
