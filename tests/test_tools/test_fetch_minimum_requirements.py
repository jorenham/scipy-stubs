import unittest.mock

import requests
from tools.fetch_minimum_requirements import get_minimum_numpy, get_minimum_python, get_pyproject


def test_package_name() -> None:
    assert get_pyproject()["project"]["name"] == "scipy-stubs"


def test_get_minimum_python() -> None:
    assert get_minimum_python().startswith("3.")


@unittest.mock.patch("requests.get")
def test_get_minimum_numpy(mocked_requests_get: unittest.mock.Mock) -> None:
    fake_response = unittest.mock.MagicMock(spec=requests.Response)
    fake_response.text = '''["project"]
requires-python = ">=3.10"
dependencies = ["numpy>=99.99.99,<100.99.99"]  # keep in sync with `min_numpy_version` in meson.build
readme = "README.rst"'''
    mocked_requests_get.return_value = fake_response
    # with unittest.mock.patch("requests.get", fake_response):
    assert get_minimum_numpy() == "99.99.99"
