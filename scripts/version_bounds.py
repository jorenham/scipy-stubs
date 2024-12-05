"""Script for the fucntions fetching the minimum versions."""

import importlib.metadata


def scipy_minimum_python() -> str:
    """Fetch the minimum Python version specified in pyproject.toml."""
    raw_version = importlib.metadata.metadata("scipy")["Requires-Python"]
    if "<" in raw_version:
        raise NotImplementedError("Version specifier with upper bound not yet supported!")
    return raw_version.replace(">=", "").replace("~=", "")


def scipy_minimum_numpy() -> str:
    """Fetch the minimum numpy version required by the current scipy package."""
    np_minimum_dep = next(req for req in importlib.metadata.requires("scipy") if req.startswith("numpy") and " extra " not in req)
    np_minimum_dep = next(ver for ver in np_minimum_dep.split(",") if ">" in ver)
    return np_minimum_dep.replace("numpy", "").replace(">=", "").replace(">", "")
