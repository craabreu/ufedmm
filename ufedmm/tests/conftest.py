"""Pytest configuration for the ufedmm tests.

The markers are registered here rather than in pyproject.toml, so that they are also
known when the tests run from an installed package (``pytest --pyargs ufedmm``).
"""


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: long statistical tests (several seconds each)"
    )
    config.addinivalue_line("markers", "gpu: runs on a CUDA or OpenCL platform")
