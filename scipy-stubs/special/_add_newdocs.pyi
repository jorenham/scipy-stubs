from typing import Final
from typing_extensions import LiteralString

docdict: Final[dict[str, str]]

def get(name: LiteralString) -> str: ...
def add_newdoc(name: LiteralString, doc: str) -> None: ...
