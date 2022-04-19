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

"""Tests for rounding_common.py"""

import pytest

from qrao import QuantumRandomAccessEncoding, RoundingContext


def test_roundingcontext_encoding_xor_var2op_provided():
    """Test that encoding or var2op must be provided in RoundingContext constructor

    (but not both).
    """
    encoding = QuantumRandomAccessEncoding()
    with pytest.raises(ValueError):
        # Can't specify both
        RoundingContext(encoding=encoding, var2op=encoding.var2op)
    with pytest.raises(ValueError):
        # Must specify one or the other
        RoundingContext()
