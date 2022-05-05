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

"""Test that the encoded Hamiltonian commutes as expected"""

import pytest

import numpy as np

from docplex.mp.model import Model

from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.problems.quadratic_program import QuadraticProgram

from qrao.encoding import QuantumRandomAccessEncoding, EncodingCommutationVerifier
from qrao.utils import get_random_maxcut_qp


def check_problem_commutation(problem: QuadraticProgram, max_vars_per_qubit: int):
    encoding = QuantumRandomAccessEncoding(max_vars_per_qubit=max_vars_per_qubit)
    encoding.encode(problem)
    verifier = EncodingCommutationVerifier(encoding)
    assert len(verifier) == 2**encoding.num_vars
    assert all(
        np.isclose(obj_val, encoded_obj_val) for _, obj_val, encoded_obj_val in verifier
    )


@pytest.mark.parametrize("max_vars_per_qubit", [1, 2, 3])
@pytest.mark.parametrize("task", ["minimize", "maximize"])
def test_one_qubit_qrac(max_vars_per_qubit, task):
    """Non-uniform weights, degree 1 terms"""
    mod = Model("maxcut")
    num_nodes = max_vars_per_qubit
    nodes = list(range(num_nodes))
    var = [mod.binary_var(name=f"x{i}") for i in nodes]
    {"minimize": mod.minimize, "maximize": mod.maximize}[task](
        mod.sum(2 * (i + 1) * var[i] for i in nodes)
    )
    problem = from_docplex_mp(mod)

    check_problem_commutation(problem, max_vars_per_qubit=max_vars_per_qubit)


@pytest.mark.parametrize("max_vars_per_qubit", [1, 2, 3])
@pytest.mark.parametrize("task", ["minimize", "maximize"])
def test_uniform_weights_degree_2(max_vars_per_qubit, task):
    # Note that the variable embedding has some qubits with 1, 2, and 3 qubits
    num_nodes = 6
    elist = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3), (1, 4), (2, 4)]
    edges = np.zeros((num_nodes, num_nodes))
    for i, j in elist:
        edges[i, j] = (i + 1) * (j + 2)

    mod = Model("maxcut")
    nodes = list(range(num_nodes))
    var = [mod.binary_var(name=f"x{i}") for i in nodes]
    {"minimize": mod.minimize, "maximize": mod.maximize}[task](
        mod.sum(
            edges[i, j] * (1 - (2 * var[i] - 1) * (2 * var[j] - 1))
            for i in nodes
            for j in nodes
        )
    )
    problem = from_docplex_mp(mod)

    check_problem_commutation(problem, max_vars_per_qubit=max_vars_per_qubit)


@pytest.mark.parametrize("max_vars_per_qubit", [1, 2, 3])
def test_random_unweighted_maxcut_problem(max_vars_per_qubit):
    problem = get_random_maxcut_qp(degree=3, num_nodes=8)
    check_problem_commutation(problem, max_vars_per_qubit=max_vars_per_qubit)


@pytest.mark.parametrize("max_vars_per_qubit", [1, 2, 3])
@pytest.mark.parametrize("task", ["minimize", "maximize"])
def test_nonuniform_weights_degree_1_and_2_terms(max_vars_per_qubit, task):
    num_nodes = 6
    elist = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3), (1, 4), (2, 5)]
    edges = np.zeros((num_nodes, num_nodes))
    for i, j in elist:
        edges[i, j] = (i + 1) * (j + 1)

    mod = Model("maxcut")
    nodes = list(range(num_nodes))
    var = [mod.binary_var(name="x" + str(i)) for i in nodes]
    expr = mod.sum(
        edges[i, j] * (var[i] + var[j] - 2 * var[i] * var[j])
        for i in nodes
        for j in nodes
    )
    expr += mod.sum(5 * (i) * var[i] for i in nodes)
    {"minimize": mod.minimize, "maximize": mod.maximize}[task](expr)
    problem = from_docplex_mp(mod)

    check_problem_commutation(problem, max_vars_per_qubit=max_vars_per_qubit)
