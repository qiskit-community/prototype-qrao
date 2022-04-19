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

"""Tests for SemideterministicRounding"""

import pytest

from qrao import (
    QuantumRandomAccessEncoding,
    RoundingContext,
    SemideterministicRounding,
)
from qrao.utils import get_random_maxcut_qp


def test_semideterministic_rounding():
    encoding = QuantumRandomAccessEncoding()
    encoding.encode(get_random_maxcut_qp(degree=3, num_nodes=6))

    rounding_scheme = SemideterministicRounding()
    tvs = [1, -1, 0, -1, 0.5, 1]
    res = rounding_scheme.round(RoundingContext(trace_values=tvs, encoding=encoding))
    assert len(res.samples) == 1
    x = res.samples[0].x.tolist()
    assert x[0] == 0
    assert x[1] == 1
    assert x[2] in (0, 1)
    assert x[3] == 1
    assert x[4] == 0
    assert x[5] == 0

    with pytest.raises(ValueError):
        # Wrong number of trace values
        rounding_scheme.round(
            RoundingContext(trace_values=[1, 0, -1], encoding=encoding)
        )
    with pytest.raises(NotImplementedError):
        # Trace values not provided
        rounding_scheme.round(RoundingContext(trace_values=None, encoding=encoding))
