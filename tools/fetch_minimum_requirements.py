import re
import subprocess
import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # TODO: remove this once 3.10 is dropped!
    import tomli as tomllib


def get_pyproject() -> dict[str, Any]:

    with Path("pyproject.toml").open("rb") as f:
        return tomllib.load(f)


def get_minimum_python() -> str:
    raw_version = get_pyproject()["project"]["requires-python"]
    if "<" in raw_version:
        raise NotImplementedError("Version specifier with upper bound not yet supported!")
    return re.sub(r"[>=~]", "", raw_version)


def get_minimum_numpy() -> str:
    """Fetch the numpy dep requirement from an output in the format:

    scipy v1.14.1
    └── numpy v2.1.3 [required: >=1.23.5, <2.3]
    """
    result = subprocess.run("uv pip tree --show-version-specifiers".split(" "), capture_output=True, check=True)
    uv_tree = result.stdout.decode()
    # TODO: why does this not work? r"^scipy.*\n└── numpy(.*)$
    numpy_dep = re.search(r"└── numpy(.*)", uv_tree).groups()
    if len(numpy_dep) > 1:
        raise NotImplementedError("Update script to include further deps!")
    return numpy_dep[0].split("required: >=")[1].split(",")[0]
