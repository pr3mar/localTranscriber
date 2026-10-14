import os

import pytest


def pytest_collection_modifyitems(config, items):
    if os.environ.get("LOCALTRANSCRIBE_GPU_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="set LOCALTRANSCRIBE_GPU_TESTS=1 inside the container to run")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
