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

"""Tests for qrao.utils"""

import pytest

from qrao.utils import get_random_maxcut_qp


def test_get_random_maxcut_qp_weight():
    """Test ``get_random_maxcut_qp()`` with each "edge case" (if statement branch) of ``weight``"""
    get_random_maxcut_qp(weight=1)
    get_random_maxcut_qp(weight=-1)
    get_random_maxcut_qp(weight=2)
    with pytest.raises(ValueError):
        get_random_maxcut_qp(weight=0)
