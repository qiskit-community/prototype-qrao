# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Pytest configuration for QRAO project"""

import pytest


# https://docs.pytest.org/en/latest/example/simple.html#control-skipping-of-tests-according-to-command-line-option


def pytest_addoption(parser):
    parser.addoption(
        "--run-backend-tests",
        action="store_true",
        default=False,
        help="run backend tests",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "backend: mark test as a backend test")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-backend-tests"):
        # --run-backend-tests given in cli: do not skip backend tests
        return
    skip_backend = pytest.mark.skip(reason="need --run-backend-tests option to run")
    for item in items:
        if "backend" in item.keywords:
            item.add_marker(skip_backend)
