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
    get_problem_encoding_state,
    get_dvars_encoding_state,
    change_to_n1p_qrac_basis,
)

# pylint: disable=protected-access


def test_qrac_unsupported_encoding():
    with pytest.raises(ValueError):
        QuantumRandomAccessEncoding(4)


def test_31p_qrac_encoding():
    encoding = QuantumRandomAccessEncoding(3)
    assert encoding.num_qubits == 0
    assert not encoding.frozen
    encoding._add_dvars([])
    assert encoding.num_qubits == 0
    with pytest.raises(ValueError):
        # Can't add the same variable twice at the same time
        encoding._add_dvars([9, 9])
    encoding._add_dvars([7, 11, 13, 17])
    assert encoding.num_qubits == 2
    with pytest.raises(ValueError):
        # Variable has already been added
        encoding._add_dvars([7])
    assert encoding.num_qubits == 2
    encoding._add_dvars([23, 27])
    assert encoding.num_qubits == 3
    assert encoding.qubit_to_dvars == [[7, 11, 13], [17], [23, 27]]
    var2q = {v: q for v, (q, _) in encoding.dvar_to_op.items()}
    assert var2q == {7: 0, 11: 0, 13: 0, 17: 1, 23: 2, 27: 2}

    m = {v: np.random.randint(2) for v in encoding.dvar_to_op}
    encoding.get_state(m)
    with pytest.raises(ValueError):
        encoding.get_state({**m, 5: 0})
    with pytest.raises(ValueError):
        encoding.get_state({**m, 7: 2})
    q2vars_modified = deepcopy(encoding.qubit_to_dvars)
    q2vars_modified[0][2] = 17  # add a duplicate (see the assertion above)
    with pytest.raises(ValueError):
        get_problem_encoding_state(m, q2vars_modified, encoding.max_dvars_per_qubit)
    q2vars_modified = deepcopy(encoding.qubit_to_dvars)
    q2vars_modified[0].append(15)  # attempt to have four variables on a qubit
    with pytest.raises(ValueError):
        get_problem_encoding_state(m, q2vars_modified, encoding.max_dvars_per_qubit)
    del m[7]
    with pytest.raises(ValueError):
        encoding.get_state(m)

    with pytest.raises(AttributeError):
        encoding.qubit_op

    encoding._add_term(1.5, 7)
    encoding._add_term(0, 7)
    with pytest.raises(KeyError):
        encoding._add_term(0, 8)

    encoding.qubit_op

    with pytest.raises(RuntimeError):
        # Collision of variables (same qubit)
        encoding.term_to_op(7, 11)
    encoding.term_to_op(7, 17)

    with pytest.raises(RuntimeError):
        encoding._add_dvars([17])

    encoding.ensure_thawed()
    encoding.freeze()
    with pytest.raises(RuntimeError):
        encoding.ensure_thawed()


def test_21p_qrac_encoding():
    encoding = QuantumRandomAccessEncoding(2)
    encoding._add_dvars([7, 11, 13])
    assert encoding.num_qubits == 2
    encoding._add_dvars([23])
    assert encoding.num_qubits == 3
    assert encoding.qubit_to_dvars == [[7, 11], [13], [23]]
    var2q = {v: q for v, (q, _) in encoding.dvar_to_op.items()}
    assert var2q == {7: 0, 11: 0, 13: 1, 23: 2}
    m = {v: np.random.randint(2) for v in encoding.dvar_to_op}
    encoding.get_state(m)


def test_111_qrac_encoding():
    encoding = QuantumRandomAccessEncoding(1)
    encoding._add_dvars([7, 11, 13])
    assert encoding.num_qubits == 3
    encoding._add_dvars([23])
    assert encoding.num_qubits == 4
    assert encoding.qubit_to_dvars == [[7], [11], [13], [23]]
    var2q = {v: q for v, (q, _) in encoding.dvar_to_op.items()}
    assert var2q == {7: 0, 11: 1, 13: 2, 23: 3}
    m = {v: np.random.randint(2) for v in encoding.dvar_to_op}
    encoding.get_state(m)


def test_qrac_encoding_from_model():
    model = Model("docplex model")
    x = model.binary_var("x")
    y = model.binary_var("y")
    model.minimize(x + 2 * y)
    problem = from_docplex_mp(model)

    encoding = QuantumRandomAccessEncoding(3)
    with pytest.raises(AttributeError):
        encoding.offset
    with pytest.raises(AttributeError):
        encoding.problem
    encoding.encode(problem)
    encoding.offset
    encoding.problem
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
    model = Model("docplex model")
    x = model.integer_var(-5, 5, "x")
    model.minimize(x)
    problem = from_docplex_mp(model)

    encoding = QuantumRandomAccessEncoding(3)
    with pytest.raises(RuntimeError):
        # It expects all variables to be binary
        encoding.encode(problem)


def test_qrac_recovery_probability():
    e = {i: QuantumRandomAccessEncoding(i) for i in (1, 2, 3)}
    assert e[1].minimum_recovery_probability == 1.0
    assert e[2].minimum_recovery_probability == pytest.approx(0.854, 0.001)
    assert e[3].minimum_recovery_probability == pytest.approx(0.789, 0.001)


def test_sense():
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

    max_encoding = QuantumRandomAccessEncoding(max_dvars_per_qubit=3)
    max_encoding.encode(get_problem(maximize=True))

    min_encoding = QuantumRandomAccessEncoding(max_dvars_per_qubit=3)
    min_encoding.encode(get_problem(maximize=False))

    assert max_encoding.qubit_op == min_encoding.qubit_op
    assert max_encoding.offset == min_encoding.offset


def test_qrac_state_prep_1q():
    with pytest.raises(TypeError):
        get_dvars_encoding_state(1, 0, 1, 0)

    for n in (1, 2, 3):
        p = QuantumRandomAccessEncoding(n).minimum_recovery_probability
        for m in itertools.product((0, 1), repeat=n):
            logical = get_dvars_encoding_state(*m)
            for i in range(n):
                op = QuantumRandomAccessEncoding.NUM_DVARS_TO_OPS[n][i]
                pauli_tv = (~logical @ op @ logical).eval()
                tv = (1 - pauli_tv) / 2
                expected_tv = p if m[i] else 1 - p
                assert tv == pytest.approx(expected_tv)


def test_undefined_basis_rotations():
    with pytest.raises(ValueError):
        change_to_n1p_qrac_basis(3, 4)  # each element should be 0, 1, 2, or 3
    with pytest.raises(ValueError):
        change_to_n1p_qrac_basis(2, 2)  # each element should be 0 or 1


def test_unassigned_qubit():
    with pytest.raises(ValueError):
        get_problem_encoding_state({}, [[]], 3)


def test_encoding_verifier_indexerror():
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
        verifier[4]
