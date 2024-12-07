# !/usr/bin/env python3
"""
Script to generate a test matrix of Python and package versions for CI testing.

This script fetches the list of supported Python versions and the versions of a
specified package available on PyPI, determines compatible combinations based
on package 'requires_python' metadata, and outputs a JSON matrix suitable for
use in CI workflows.

Features:
- Generalized to work with any package (not just NumPy).
- Caches HTTP requests to reduce network calls.
- Prevents code duplication by using reusable functions.
- Includes thorough doctests and documentation.

Usage:
    python generate_matrix.py [PACKAGE_NAME]

Example:
    python generate_matrix.py numpy

"""

import dataclasses
import importlib.metadata
import json
import sys
import urllib.error
import urllib.request
from functools import lru_cache

# Constants for caching
CACHE_SIZE = 128  # Adjust as needed


@dataclasses.dataclass(frozen=True, eq=True, order=True)
class Version:
    major: int
    minor: int
    micro: int

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.micro}"

    @property
    def minor_version(self) -> tuple[int, int]:
        return self.major, self.minor

    @classmethod
    def from_str(cls, version_str: str) -> "Version":
        if version_str.count(".") == 1:
            version_str += ".0"
        return cls(*map(int, version_str.strip().split(".")))


def get_minimum_python_version(package: str) -> Version:
    """
    Fetch the minimum Python version specified in pyproject.toml.

    Returns:
        str: The minimum required version of the package.

    Examples:
        >>> get_minimum_python_version()  # doctest: +ELLIPSIS
        Version(major=3, ...)
    """
    raw_version = importlib.metadata.metadata(package)["Requires-Python"]
    if "<" in raw_version:
        raise NotImplementedError("Version specifier with upper bound not yet supported!")

    return Version.from_str(raw_version.replace(">=", "").replace("~=", ""))


def get_minimum_version(package: str, dependency: str) -> Version:
    """
    Fetch the minimum dependency version required by the given package.

    Args:
        package (str): The package name ('scipy' or package name).
        dependency (str): The dependency name ('numpy' or dependency name).

    Returns:
        str: The minimum required version of the package.

    Examples:
        >>> get_minimum_version("scipy", "numpy")  # doctest: +ELLIPSIS
        Version(major=1, ...)
    """
    np_minimum_dep = next(
        req for req in importlib.metadata.requires(package) if req.startswith(dependency) and " extra " not in req
    )
    np_minimum_dep = next(ver for ver in np_minimum_dep.split(",") if ">" in ver)
    return Version.from_str(np_minimum_dep.replace(dependency, "").replace(">=", "").replace(">", ""))


def get_python_versions(
    min_version: Version | None = None,
    max_version: Version | None = None,
) -> list[Version]:
    """
    Fetch the list of available Python versions from the GitHub Actions Python Versions Manifest.

    Args:
        min_version (Version): The minimum version to include.
        max_version (Version): The maximum version to include.
        include_prerelease (bool): Whether to include pre-release versions.

    Returns:
        list[Version]: A list of available Python versions.

    Raises:
        urllib.error.URLError: If fetching data fails.

    Examples:
        >>> versions = get_python_versions()
        >>> versions[0] >= Version(3, 6, 0)
        True
    """
    data = fetch_json("https://raw.githubusercontent.com/actions/python-versions/main/versions-manifest.json")

    versions = {}

    for release in data[::-1]:
        try:
            version = Version.from_str(release["version"])
        except ValueError:
            # Skip pre-release versions
            continue

        if min_version and version < min_version:
            continue
        if max_version and version > max_version:
            continue

        versions[version.minor_version] = version

    return sorted(versions.values())


@lru_cache(maxsize=CACHE_SIZE)
def fetch_json(url: str) -> dict | list:
    """
    Fetch JSON data from a URL with caching.

    Args:
        url (str): The URL to fetch.

    Returns:
        Union[Dict, List]: The parsed JSON data.

    Raises:
        urllib.error.URLError: If fetching data fails.

    Examples:
        >>> data = fetch_json("https://pypi.org/pypi/numpy/json")
        >>> isinstance(data, dict)
        True
    """
    try:
        with urllib.request.urlopen(url) as response:  # noqa: S310
            return json.loads(response.read().decode())
    except urllib.error.URLError:
        sys.exit(1)


def get_package_versions(package_name: str, min_version: Version) -> list[str]:
    """
    Fetch available package versions from PyPI starting from the minimum version.

    Args:
        package_name (str): The name of the package on PyPI.
        min_version_str (str): The minimum version string.

    Returns:
        List[str]: A sorted list of package versions as strings.

    Examples:
        >>> versions = get_package_versions("numpy", "1.21")
        >>> "1.21.0" in versions
        True
    """
    data = fetch_json(f"https://pypi.org/pypi/{package_name}/json")
    releases = data.get("releases", {})
    versions = {}
    for version_str in releases:
        if any(c in version_str for c in ["a", "b", "rc", "dev"]):
            continue  # Skip pre-releases
        try:
            version = Version.from_str(version_str)
            if version >= min_version:
                versions[version.minor_version] = version
        except ValueError:
            continue  # Skip invalid versions

    return sorted(versions.values())


def main() -> None:
    """
    Main function to generate and output the test matrix.

    Reads package names from command-line arguments and outputs a JSON matrix
    of compatible Python and package versions.
    """
    package_name = "scipy"
    dependency_name = "numpy"

    min_python_version = get_minimum_python_version(package_name)
    python_versions = get_python_versions(min_python_version)

    min_package_version = max(
        get_minimum_version(package_name, dependency_name),
        # Due to the distutils requirement we cannot use old numpy versions
        Version(1, 26, 0),
    )

    package_versions = get_package_versions(dependency_name, min_package_version)

    matrix = {
        "python": [str(v) for v in python_versions],
        "numpy": [str(v) for v in package_versions],
    }
    json.dump(matrix, fp=sys.stdout)


if __name__ == "__main__":
    main()
