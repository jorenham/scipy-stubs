import importlib.metadata
import json
import sys
import urllib.error
import urllib.request
from functools import lru_cache
import typing

from packaging.specifiers import SpecifierSet
from packaging.version import Version, parse

# Constants for caching
CACHE_SIZE = 128  # Adjust as needed


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


@lru_cache(maxsize=CACHE_SIZE)
def get_minimum_python_version(package: str) -> Version:
    """
    Fetch the minimum Python version specified in the package's metadata.

    Returns:
        Version: The minimum required Python version.

    """
    raw_version = importlib.metadata.metadata(package)["Requires-Python"]
    if "<" in raw_version:
        raise NotImplementedError("Version specifier with upper bound not yet supported!")

    return parse(raw_version.replace(">=", "").replace("~=", ""))


def get_minimum_version(package: str, dependency: str) -> Version:
    """
    Fetch the minimum dependency version required by the given package.

    Args:
        package (str): The package name (e.g., 'scipy').
        dependency (str): The dependency name (e.g., 'numpy').

    Returns:
        Version: The minimum required version of the dependency.

    """
    requirements = importlib.metadata.requires(package)
    if not requirements:
        error = f"No requirements for {package}"
        raise ValueError(error)

    np_minimum_dep = next(req for req in requirements if req.startswith(dependency) and " extra " not in req)
    np_minimum_dep = next(ver for ver in np_minimum_dep.split(",") if ">" in ver)
    return parse(np_minimum_dep.replace(dependency, "").replace(">=", "").replace(">", ""))


def get_python_versions(
    min_version: Version | None = None,
    max_version: Version | None = None,
    pre_releases: bool = False,
) -> list[Version]:
    """
    Fetch the list of available Python versions from GitHub Actions' Python Versions Manifest.

    Args:
        min_version (Version): The minimum version to include.
        max_version (Version): The maximum version to include.

    Returns:
        list[Version]: A list of available Python versions.

    Raises:
        urllib.error.URLError: If fetching data fails.

    """
    data: list[Release] = typing.cast(
        "list[Release]", fetch_json("https://raw.githubusercontent.com/actions/python-versions/main/versions-manifest.json")
    )

    versions: dict[tuple[int, int], Version] = {}

    for release in data[::-1]:
        try:
            version = parse(release["version"])
        except ValueError:
            # Skip invalid version strings
            continue

        if version.is_prerelease and not pre_releases:
            continue

        if min_version and version < min_version:
            continue
        if max_version and version > max_version:
            continue

        versions[version.major, version.minor] = version

    return sorted(versions.values())


@lru_cache(maxsize=CACHE_SIZE)
def fetch_json(url: str) -> dict[str, typing.Any] | list[typing.Any]:
    """
    Fetch JSON data from a URL with caching.

    Args:
        url (str): The URL to fetch.

    Returns:
        dict: The parsed JSON data.

    Raises:
        urllib.error.URLError: If fetching data fails.

    """
    try:
        with urllib.request.urlopen(url) as response:  # noqa: S310
            return json.loads(response.read())
    except urllib.error.URLError:
        sys.exit(1)


def get_package_versions(package_name: str, min_version: Version) -> dict[Version, str]:
    """
    Fetch available package versions from PyPI starting from the minimum version,
    along with their 'requires_python'.

    Args:
        package_name (str): The name of the package on PyPI.
        min_version (Version): The minimum version to include.

    Returns:
        dict[Version, str]: A mapping from package versions to their 'requires_python'.

    """
    data: PackageVersions = typing.cast(
        "PackageVersions",
        typing.cast(
            "object",
            fetch_json(
                f"https://pypi.org/pypi/{package_name}/json",
            ),
        ),
    )

    releases = data.get("releases", {})
    versions: dict[Version, str] = {}
    for version_str in releases:
        version = parse(version_str)
        if version < min_version:
            continue
        # Get 'requires_python' for this version
        release_files = releases[version_str]
        requires_python = None
        for file_info in release_files:
            if requires_python := file_info.get("requires_python"):
                break

        if requires_python is None:
            error = f"No 'requires_python' found for {package_name} {version}"
            raise RuntimeError(error)

        versions[version] = requires_python

    return versions


def main() -> None:
    """
    Main function to generate and output the test matrix.

    Outputs a JSON matrix of compatible Python and package versions.
    """
    package_name = "scipy"
    dependency_name = "numpy"

    min_python_version = get_minimum_python_version(package_name)
    python_versions = get_python_versions(min_python_version)

    package_versions = get_package_versions(
        dependency_name,
        get_minimum_version(package_name, dependency_name),
    )

    include: list[dict[str, str]] = []

    for python_version in python_versions:
        for package_version, requires_python in package_versions.items():
            specifier_set = SpecifierSet(requires_python)
            if python_version in specifier_set:
                include.append(
                    {
                        "python": str(python_version),
                        dependency_name: str(package_version),
                    }
                )

    matrix = {"include": include}

    json.dump(matrix, indent=4, fp=sys.stdout)


if __name__ == "__main__":
    main()
