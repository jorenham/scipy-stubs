# ruff: noqa: TRY003
import importlib.metadata
import json
import sys
import urllib.error
import urllib.request
from functools import lru_cache
import typing

from packaging.specifiers import SpecifierSet
from packaging.version import Version, parse

# Constants
CACHE_SIZE = 128  # Adjust as needed
PACKAGE_NAME = "scipy"
DEPENDENCY_NAME = "numpy"
MINIMUM_DEPENDENCY_VERSION_FOR_PYTHON_3_12 = Version("1.26.0")


class FileInfo(typing.TypedDict, total=False):
    filename: str
    arch: str
    platform: str
    platform_version: str | None
    download_url: str
    requires_python: str


class Release(typing.TypedDict):
    version: str
    stable: bool
    release_url: str
    files: list[FileInfo]


class PackageVersions(typing.TypedDict, total=False):
    releases: dict[str, list[FileInfo]]


@lru_cache(maxsize=CACHE_SIZE)  # type: ignore[no-any-expr]
def get_package_minimum_python_version(package: str) -> Version:
    """
    Get the minimum Python version required by the specified package.

    Args:
        package (str): The name of the package.

    Returns:
        Version: The minimum Python version required by the package as specified
                 in its metadata.

    Raises:
        KeyError: If 'Requires-Python' is not specified in the package metadata.
        NotImplementedError: If the version specifier contains an upper bound.

    """
    raw_version = importlib.metadata.metadata(package)["Requires-Python"]
    if "<" in raw_version:
        raise NotImplementedError("Version specifier with upper bound not yet supported!")

    return parse(raw_version.replace(">=", "").replace("~=", ""))


def get_dependency_minimum_version(package: str, dependency: str) -> Version:
    """
    Get the minimum version of a dependency required by the specified package.

    Args:
        package (str): The name of the package.
        dependency (str): The name of the dependency.

    Returns:
        Version: The minimum required version of the dependency.

    Raises:
        ValueError: If the package does not list any requirements or the dependency is not found.

    """
    requirements = importlib.metadata.requires(package)
    if not requirements:
        raise ValueError(f"No requirements for {package}")

    try:
        # Find the first requirement that matches the dependency and is not an extra
        dependency_req = next(req for req in requirements if req.startswith(dependency) and " extra " not in req)
    except StopIteration as e:
        raise ValueError(f"Dependency {dependency} not found in requirements for {package}") from e

        # Extract the version specifier (e.g., ">=1.21.0")
    version_specifier = next(
        (ver for ver in dependency_req.split(",") if ">" in ver),
        None,
    )
    if version_specifier is None:
        raise ValueError(f"No version specifier found for dependency {dependency} in {package}")

    # Remove dependency name and comparison operator to get the version
    version_str = version_specifier.replace(dependency, "").replace(">=", "").replace(">", "")
    return parse(version_str)


def get_available_python_versions(
    min_version: Version | None = None,
    max_version: Version | None = None,
    pre_releases: bool = False,
) -> list[Version]:
    """
    Get a list of available Python versions from GitHub Actions' Python Versions Manifest.

    Args:
        min_version (Version | None): The minimum Python version to include in the list.
        max_version (Version | None): The maximum Python version to include in the list.
        pre_releases (bool): Whether to include pre-release versions.

    Returns:
        list[Version]: A list of available Python versions satisfying the specified criteria.

    Raises:
        urllib.error.URLError: If fetching data fails.

    """
    data: list[Release] = typing.cast(
        "list[Release]",
        fetch_json("https://raw.githubusercontent.com/actions/python-versions/main/versions-manifest.json"),
    )

    versions: dict[tuple[int, int], Version] = {}

    for release in data[::-1]:
        version = parse(release["version"])

        if version.is_prerelease and not pre_releases:
            continue

        if min_version and version < min_version:
            continue
        if max_version and version > max_version:
            continue

        versions[version.major, version.minor] = version

    return sorted(versions.values())


@lru_cache(maxsize=CACHE_SIZE)  # type: ignore[no-any-expr]
def fetch_json(url: str) -> object:
    """
    Fetch JSON data from a URL with caching.

    Args:
        url (str): The URL to fetch.

    Returns:
        dict[str, typing.Any] | list[typing.Any]: The parsed JSON data.

    Raises:
        urllib.error.URLError: If fetching data fails.

    """
    try:
        with urllib.request.urlopen(url) as response:  # type: ignore[no-any-expr]  # noqa: S310
            return typing.cast("object", json.loads(response.read()))  # type: ignore[no-any-expr]
    except urllib.error.URLError:
        sys.exit(1)


def get_available_package_versions(package_name: str, min_version: Version) -> dict[Version, str]:
    """
    Get available package versions from PyPI starting from the specified minimum version,
    along with their 'requires_python' specifiers.

    Args:
        package_name (str): The name of the package on PyPI.
        min_version (Version): The minimum version to include.

    Returns:
        dict[Version, str]: A mapping from package versions to their 'requires_python' specifier.

    Raises:
        RuntimeError: If no 'requires_python' is found for a package version.

    """
    data: PackageVersions = typing.cast(
        "PackageVersions",
        fetch_json(f"https://pypi.org/pypi/{package_name}/json"),
    )

    releases = data.get("releases", {})
    versions: dict[Version, str] = {}
    for version_str, release_files in releases.items():
        version = parse(version_str)
        if version < min_version:
            continue
            # Get 'requires_python' for this version
        requires_python = None
        for file_info in release_files:
            if "requires_python" in file_info:
                requires_python = file_info["requires_python"]
                if requires_python:
                    break

        if requires_python is None:
            raise RuntimeError(f"No 'requires_python' found for {package_name} {version}")

        versions[version] = requires_python

    return versions


def main() -> None:
    """
    Main function to generate and output the test matrix.

    This function fetches the minimum required Python version for the package
    ('PACKAGE_NAME'), gets the list of available Python versions starting from
    that minimum version, and fetches the available versions of the dependency
    ('DEPENDENCY_NAME') starting from the minimum version required by the package.

    It then computes a matrix of combinations of Python versions and dependency
    versions that are compatible according to the 'requires_python' specifiers.

    Outputs:
        A JSON object representing the test matrix, printed to stdout.

    """
    # Get the minimum required Python version for the package
    min_python_version = get_package_minimum_python_version(PACKAGE_NAME)

    # Get the available Python versions starting from the minimum version
    python_versions = get_available_python_versions(min_python_version)

    # Get the minimum required version of the dependency for the package
    min_dependency_version = get_dependency_minimum_version(PACKAGE_NAME, DEPENDENCY_NAME)

    # Get the available versions of the dependency starting from the minimum version
    package_versions = get_available_package_versions(
        DEPENDENCY_NAME,
        min_dependency_version,
    )

    include: list[dict[str, str]] = []

    for package_version, requires_python in package_versions.items():
        specifier_set = SpecifierSet(requires_python)

        for python_version in python_versions:
            if python_version in specifier_set:
                # Skip incompatible combinations
                if python_version >= Version("3.12") and package_version < MINIMUM_DEPENDENCY_VERSION_FOR_PYTHON_3_12:
                    continue

                include.append(
                    {
                        "python": str(python_version),
                        DEPENDENCY_NAME: str(package_version),
                    }
                )

    matrix = {"include": include}

    json.dump(matrix, indent=4, fp=sys.stdout)


if __name__ == "__main__":
    main()
