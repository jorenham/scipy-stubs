import re
import sys
from pathlib import Path
from typing import Any

import requests

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
    scipy_group = get_pyproject()["dependency-groups"]["scipy"]
    scipy_version = next(dep for dep in scipy_group if dep.startswith("scipy==")).replace("scipy==", "")

    response = requests.get(
        f"https://raw.githubusercontent.com/scipy/scipy/refs/tags/v{scipy_version}/pyproject.toml",
        timeout=10,
    )
    scipy_pyproject = tomllib.loads(response.text)

    numpy_minimum_dep = next(dep for dep in scipy_pyproject["project"]["dependencies"] if dep.startswith("numpy>="))
    numpy_minimum_dep = numpy_minimum_dep.replace("numpy>=", "")
    if ",<" in numpy_minimum_dep:  # Remove any upper dep if specified
        numpy_minimum_dep = numpy_minimum_dep.split(",")[0]

    return numpy_minimum_dep
