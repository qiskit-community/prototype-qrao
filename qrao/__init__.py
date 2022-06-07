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

"""
QRAO classes and functions
==========================

Quantum Random Access Optimization.

.. autosummary::
    :toctree: ../stubs/

    encoding
    rounding_common
    SemideterministicRounding
    MagicRounding
    QuantumRandomAccessOptimizer
    utils
"""

from importlib_metadata import PackageNotFoundError
from importlib_metadata import version as metadata_version

from .encoding import QuantumRandomAccessEncoding
from .magic_rounding import MagicRounding, MagicRoundingResult
from .quantum_random_access_optimizer import (
    QuantumRandomAccessOptimizationResult,
    QuantumRandomAccessOptimizer,
)
from .rounding_common import RoundingContext, RoundingResult, RoundingScheme
from .semideterministic_rounding import (
    SemideterministicRounding,
    SemideterministicRoundingResult,
)

try:
    __version__ = metadata_version("prototype_template")
except PackageNotFoundError:  # pragma: no cover
    # package is not installed
    pass
