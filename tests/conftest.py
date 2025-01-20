"""
Re-useable fixtures etc. for tests

See https://docs.pytest.org/en/7.1.x/reference/fixtures.html#conftest-py-sharing-fixtures-across-multiple-files
"""

from __future__ import annotations

import importlib

import pytest

import continuous_timeseries.pandas_accessors


def pytest_report_header(config):
    dep_info = []
    for dep in [
        "attrs",
        "continuous_timeseries",
        "matplotlib",
        "numpy",
        "pandas",
        "pint",
        "scipy",
        "tqdm",
    ]:
        try:
            dep_version = importlib.import_module(dep).__version__
            dep_info.append(f"{dep}: {dep_version}")
        except ImportError:
            dep_info.append(f"{dep}: not installed")

    return "\n".join(dep_info)


@pytest.fixture()
def setup_pandas_accessor() -> None:
    pd = pytest.importorskip("pandas")

    # Not parallel safe, but good enough
    continuous_timeseries.pandas_accessors.register_pandas_accessor()

    yield None

    # Surprising and a bit annoying that there isn't a safer way to do this
    pd.Series._accessors.discard("ct")
    if hasattr(pd.Series, "ct"):
        del pd.Series.ct
