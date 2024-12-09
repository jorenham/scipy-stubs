"""Script for functions fetching the minimum versions required by scipy."""

import importlib.metadata
import sys


def scipy_minimum_python() -> str:
    """Fetch the minimum Python version specified in pyproject.toml."""
    raw_version = importlib.metadata.metadata("scipy")["Requires-Python"]
    if "<" in raw_version:
        raise NotImplementedError("Version specifier with upper bound not yet supported!")
    return raw_version.replace(">=", "").replace("~=", "")


def scipy_minimum_numpy() -> str:
    """Fetch the minimum numpy version required by the current scipy package."""
    reqs = importlib.metadata.requires("scipy") or []
    np_minimum_dep = next(req for req in reqs if req.startswith("numpy") and " extra " not in req)
    np_minimum_dep = next(ver for ver in np_minimum_dep.split(",") if ">" in ver)
    return np_minimum_dep.replace("numpy", "").replace(">=", "").replace(">", "")


def main() -> None:
    match sys.argv:
        case [_, "python"]:
            print(scipy_minimum_python())  # noqa: T201
        case [_, "numpy"]:
            print(scipy_minimum_numpy())  # noqa: T201
        case _ as badargs:
            print(f"Usage: {badargs[0]} [python|numpy]", file=sys.stderr)  # noqa: T201
            sys.exit(1)


if __name__ == "__main__":
    main()
