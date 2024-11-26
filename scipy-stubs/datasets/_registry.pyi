from typing import Final, Literal as L, TypedDict, type_check_only
from typing_extensions import LiteralString

@type_check_only
class _MethodRegistry(TypedDict):
    ascent: list[L["ascent.dat"]]
    electrocardiogram: list[L["ecg.dat"]]
    face: list[L["face.dat"]]

_DataRegistry = TypedDict("_DataRegistry", {"ascent.dat": LiteralString, "ecg.dat": LiteralString, "face.dat": LiteralString})

registry: Final[_DataRegistry] = ...
registry_urls: Final[_DataRegistry] = ...
method_files_map: Final[_MethodRegistry] = ...
