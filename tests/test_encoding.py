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

"""Tests for qrao.encoding"""

from copy import deepcopy
import itertools

import pytest

import numpy as np

from qiskit_optimization.translators import from_docplex_mp
from docplex.mp.model import Model

from qrao.encoding import (
    QuantumRandomAccessEncoding,
    EncodingCommutationVerifier,
    qrac_state_prep_multiqubit,
    qrac_state_prep_1q,
    z_to_31p_qrac_basis_circuit,
    z_to_21p_qrac_basis_circuit,
)

# pylint: disable=protected-access


def test_qrac_unsupported_encoding():
    """Test that exception is raised if ``max_vars_per_qubit`` is invalid"""
    with pytest.raises(ValueError):
        QuantumRandomAccessEncoding(4)
    with pytest.raises(ValueError):
        QuantumRandomAccessEncoding(0)
    with pytest.raises(TypeError):
        QuantumRandomAccessEncoding(1.0)


def test_31p_qrac_encoding():  # pylint: disable=too-many-statements
    """Test (3,1,p) QRAC"""
    encoding = QuantumRandomAccessEncoding(3)
    assert encoding.num_qubits == 0
    assert not encoding.frozen
    encoding._add_variables([])
    assert encoding.num_qubits == 0
    with pytest.raises(ValueError):
        # Can't add the same variable twice at the same time
        encoding._add_variables([9, 9])
    encoding._add_variables([7, 11, 13, 17])
    assert encoding.num_qubits == 2
    with pytest.raises(ValueError):
        # Variable has already been added
        encoding._add_variables([7])
    assert encoding.num_qubits == 2
    encoding._add_variables([23, 27])
    assert encoding.num_qubits == 3
    assert encoding.q2vars == [[7, 11, 13], [17], [23, 27]]
    var2q = {v: q for v, (q, _) in encoding.var2op.items()}
    assert var2q == {7: 0, 11: 0, 13: 0, 17: 1, 23: 2, 27: 2}

    m = {v: np.random.randint(2) for v in encoding.var2op}
    encoding.state_prep(m)
    with pytest.raises(ValueError):
        encoding.state_prep({**m, 5: 0})
    with pytest.raises(ValueError):
        encoding.state_prep({**m, 7: 2})
    q2vars_modified = deepcopy(encoding.q2vars)
    q2vars_modified[0][2] = 17  # add a duplicate (see the assertion above)
    with pytest.raises(ValueError):
        qrac_state_prep_multiqubit(m, q2vars_modified, encoding.max_vars_per_qubit)
    q2vars_modified = deepcopy(encoding.q2vars)
    q2vars_modified[0].append(15)  # attempt to have four variables on a qubit
    with pytest.raises(ValueError):
        qrac_state_prep_multiqubit(m, q2vars_modified, encoding.max_vars_per_qubit)
    del m[7]
    with pytest.raises(ValueError):
        encoding.state_prep(m)

    with pytest.raises(AttributeError):
        encoding.qubit_op  # pylint: disable=pointless-statement

    encoding._add_term(1.5, 7)
    encoding._add_term(0, 7)
    with pytest.raises(KeyError):
        encoding._add_term(0, 8)

    encoding.qubit_op  # pylint: disable=pointless-statement

    with pytest.raises(RuntimeError):
        # Collision of variables (same qubit)
        encoding.term2op(7, 11)
    encoding.term2op(7, 17)

    with pytest.raises(RuntimeError):
        encoding._add_variables([17])

    encoding.ensure_thawed()
    encoding.freeze()
    with pytest.raises(RuntimeError):
        encoding.ensure_thawed()


def test_21p_qrac_encoding():
    """Test (2,1,p) QRAC"""
    encoding = QuantumRandomAccessEncoding(2)
    encoding._add_variables([7, 11, 13])
    assert encoding.num_qubits == 2
    encoding._add_variables([23])
    assert encoding.num_qubits == 3
    assert encoding.q2vars == [[7, 11], [13], [23]]
    var2q = {v: q for v, (q, _) in encoding.var2op.items()}
    assert var2q == {7: 0, 11: 0, 13: 1, 23: 2}
    m = {v: np.random.randint(2) for v in encoding.var2op}
    encoding.state_prep(m)


def test_111_qrac_encoding():
    """Test (1,1,1) QRAC"""
    encoding = QuantumRandomAccessEncoding(1)
    encoding._add_variables([7, 11, 13])
    assert encoding.num_qubits == 3
    encoding._add_variables([23])
    assert encoding.num_qubits == 4
    assert encoding.q2vars == [[7], [11], [13], [23]]
    var2q = {v: q for v, (q, _) in encoding.var2op.items()}
    assert var2q == {7: 0, 11: 1, 13: 2, 23: 3}
    m = {v: np.random.randint(2) for v in encoding.var2op}
    encoding.state_prep(m)


def test_qrac_encoding_from_model():
    """Test QRAC encoding from DOcplex model"""
    model = Model("docplex model")
    x = model.binary_var("x")
    y = model.binary_var("y")
    model.minimize(x + 2 * y)
    problem = from_docplex_mp(model)

    encoding = QuantumRandomAccessEncoding(3)
    with pytest.raises(AttributeError):
        encoding.offset  # pylint: disable=pointless-statement
    with pytest.raises(AttributeError):
        encoding.problem  # pylint: disable=pointless-statement
    encoding.encode(problem)
    encoding.offset  # pylint: disable=pointless-statement
    encoding.problem  # pylint: disable=pointless-statement
    assert encoding.num_qubits == 1
    assert encoding.compression_ratio == 2

    with pytest.raises(RuntimeError):
        # Can't encode an object that already has already been used
        encoding.encode(problem)

    model.add_constraint(x + y == 1)
    problem2 = from_docplex_mp(model)
    encoding2 = QuantumRandomAccessEncoding(3)
    with pytest.raises(RuntimeError):
        # It expects an unconstrained problem
        encoding2.encode(problem2)


def test_qrac_encoding_from_invalid_model2():
    """Test QRAC encoding from DOcplex model that is not a QUBO"""
    model = Model("docplex model")
    x = model.integer_var(-5, 5, "x")
    model.minimize(x)
    problem = from_docplex_mp(model)

    encoding = QuantumRandomAccessEncoding(3)
    with pytest.raises(RuntimeError):
        # It expects all variables to be binary
        encoding.encode(problem)


def test_qrac_recovery_probability():
    """Test that each QRAC returns the correct recovery probability"""
    e = {i: QuantumRandomAccessEncoding(i) for i in (1, 2, 3)}
    assert e[1].minimum_recovery_probability == 1.0
    assert e[2].minimum_recovery_probability == pytest.approx(0.854, 0.001)
    assert e[3].minimum_recovery_probability == pytest.approx(0.789, 0.001)


def test_sense():
    """Test that minimization and maximization problems relate to each other as expected"""

    def get_problem(maximize=True):
        # Load small reference problem (includes a self-loop)
        elist = [
            (0, 1),
            (0, 4),
            (0, 3),
            (1, 2),
            (1, 5),
            (2, 3),
            (2, 4),
            (4, 5),
            (5, 3),
            (1, 1),
        ]
        num_nodes = 6
        mod = Model("maxcut")
        nodes = list(range(num_nodes))
        var = [mod.binary_var(name="x" + str(i)) for i in nodes]
        if maximize:
            mod.maximize(
                mod.sum((var[i] + var[j] - 2 * var[i] * var[j]) for i, j in elist)
            )
        else:
            mod.minimize(
                mod.sum(
                    (-1 * (var[i] + var[j] - 2 * var[i] * var[j])) for i, j in elist
                )
            )
        problem = from_docplex_mp(mod)
        return problem

    max_encoding = QuantumRandomAccessEncoding(max_vars_per_qubit=3)
    max_encoding.encode(get_problem(maximize=True))

    min_encoding = QuantumRandomAccessEncoding(max_vars_per_qubit=3)
    min_encoding.encode(get_problem(maximize=False))

    assert max_encoding.qubit_op == min_encoding.qubit_op
    assert max_encoding.offset == min_encoding.offset


def test_qrac_state_prep_1q():
    """Test each possble QRAC state preparation on a single qubit"""
    with pytest.raises(TypeError):
        qrac_state_prep_1q(1, 0, 1, 0)

    for n in (1, 2, 3):
        p = QuantumRandomAccessEncoding(n).minimum_recovery_probability
        for m in itertools.product((0, 1), repeat=n):
            logical = qrac_state_prep_1q(*m)
            for i in range(n):
                op = QuantumRandomAccessEncoding.OPERATORS[n - 1][i]
                pauli_tv = (~logical @ op @ logical).eval()
                tv = (1 - pauli_tv) / 2
                expected_tv = p if m[i] else 1 - p
                assert tv == pytest.approx(expected_tv)


def test_undefined_basis_rotations():
    """Test that undefined basis rotations raise ``ValueError``"""
    with pytest.raises(ValueError):
        z_to_31p_qrac_basis_circuit([4])  # each element should be 0, 1, 2, or 3
    with pytest.raises(ValueError):
        z_to_21p_qrac_basis_circuit([2])  # each element should be 0 or 1


def test_unassigned_qubit():
    """Test that qubit with no decision variables assigned to it raises error"""
    with pytest.raises(ValueError):
        qrac_state_prep_multiqubit({}, [[]], 3)


def test_encoding_verifier_indexerror():
    """Test that ``EncodingCommutationVerifier`` raises ``IndexError`` if indexed out of range"""
    model = Model("docplex model")
    x = model.binary_var("x")
    y = model.binary_var("y")
    model.minimize(x + 2 * y)
    problem = from_docplex_mp(model)

    encoding = QuantumRandomAccessEncoding(3)
    encoding.encode(problem)

    verifier = EncodingCommutationVerifier(encoding)
    assert len(verifier) == 4
    with pytest.raises(IndexError):
        verifier[4]  # pylint: disable=pointless-statement
