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

"""Tests for QuantumRandomAccessOptimizer."""

from unittest import TestCase

from qiskit.utils import QuantumInstance
from qiskit.algorithms.minimum_eigensolvers import (
    VQE,
    QAOA,
    NumPyMinimumEigensolver,
    MinimumEigensolverResult,
)
from qiskit.circuit.library import RealAmplitudes
from qiskit.algorithms.optimizers import SPSA
from qiskit.primitives import Estimator, Sampler
from qiskit_aer import Aer

from qiskit_optimization.translators import from_docplex_mp
from docplex.mp.model import Model

from qrao import (
    QuantumRandomAccessOptimizer,
    QuantumRandomAccessEncoding,
    RoundingContext,
    MagicRounding,
    RoundingResult,
)
from qrao.utils import get_random_maxcut_qp


class TestQuantumRandomAccessOptimizer(TestCase):
    """Test QuantumRandomAccessOptimizer."""

    def setUp(self):
        # Load a problem to test out
        self.problem = get_random_maxcut_qp(degree=3, num_nodes=6)
        self.encoding = QuantumRandomAccessEncoding(max_vars_per_qubit=3)
        self.encoding.encode(self.problem)
        self.assertEqual(self.encoding.num_qubits, self.encoding.qubit_op.num_qubits)
        self.ansatz = RealAmplitudes(self.encoding.num_qubits)  # for VQE

    def test_solve_relaxed_vqe(self):
        """Test QuantumRandomAccessOptimizer with VQE."""
        estimator = Estimator(options={"shots": 100})
        vqe = VQE(
            ansatz=self.ansatz,
            optimizer=SPSA(maxiter=1, learning_rate=0.01, perturbation=0.1),
            estimator=estimator,
        )

        qrao = QuantumRandomAccessOptimizer(
            encoding=self.encoding, min_eigen_solver=vqe
        )
        relaxed_results, rounding_context = qrao.solve_relaxed()
        self.assertIsInstance(relaxed_results, MinimumEigensolverResult)
        self.assertIsInstance(rounding_context, RoundingContext)

    def test_solve_relaxed_qaoa(self):
        """Test QuantumRandomAccessOptimizer with QAOA."""
        sampler = Sampler(options={"shots": 100})
        qaoa = QAOA(
            optimizer=SPSA(maxiter=1, learning_rate=0.01, perturbation=0.1),
            sampler=sampler,
            mixer=self.encoding.qubit_op,
        )
        qrao = QuantumRandomAccessOptimizer(
            encoding=self.encoding, min_eigen_solver=qaoa
        )
        relaxed_results, rounding_context = qrao.solve_relaxed()
        self.assertIsInstance(relaxed_results, MinimumEigensolverResult)
        self.assertIsInstance(rounding_context, RoundingContext)

    def test_solve_relaxed_numpy(self):
        """Test QuantumRandomAccessOptimizer with NumPyMinimumEigensolver."""
        np_solver = NumPyMinimumEigensolver()
        qrao = QuantumRandomAccessOptimizer(
            encoding=self.encoding, min_eigen_solver=np_solver
        )
        relaxed_results, rounding_context = qrao.solve_relaxed()
        self.assertIsInstance(relaxed_results, MinimumEigensolverResult)
        self.assertIsInstance(rounding_context, RoundingContext)

    def test_different_problem(self):
        """Test passing a different problem to solve() than the encoding."""
        np_solver = NumPyMinimumEigensolver()
        qrao = QuantumRandomAccessOptimizer(
            encoding=self.encoding, min_eigen_solver=np_solver
        )
        other_problem = from_docplex_mp(Model("docplex model"))
        self.assertEqual(qrao.get_compatibility_msg(self.problem), "")
        self.assertNotEqual(qrao.get_compatibility_msg(other_problem), "")
        with self.assertRaises(ValueError):
            qrao.solve(other_problem)

    def test_require_aux_operator_support(self):
        """Test than the eigensolver is tested for aux operator support

        If aux operators are not supported, a TypeError should be raised.
        """

        class ModifiedVQE(VQE):
            """Modified VQE method without aux operator support

            No existing eigensolver seems to be without aux operator support, so
            we make this one that claims to lack it.
            """

            @classmethod
            def supports_aux_operators(cls) -> bool:
                return False

        estimator = Estimator(options={"shots": 100})
        vqe = ModifiedVQE(
            ansatz=self.ansatz,
            optimizer=SPSA(maxiter=1, learning_rate=0.01, perturbation=0.1),
            estimator=estimator,
        )
        with self.assertRaises(TypeError):
            QuantumRandomAccessOptimizer(encoding=self.encoding, min_eigen_solver=vqe)

    def test_solve_without_args(self):
        """Test solve() without arguments"""
        estimator = Estimator(options={"shots": 100})
        rounding_qi = QuantumInstance(
            backend=Aer.get_backend("aer_simulator"), shots=100
        )

        vqe = VQE(
            ansatz=self.ansatz,
            optimizer=SPSA(maxiter=1, learning_rate=0.01, perturbation=0.1),
            estimator=estimator,
        )
        rounding_scheme = MagicRounding(quantum_instance=rounding_qi)

        qrao = QuantumRandomAccessOptimizer(
            encoding=self.encoding,
            min_eigen_solver=vqe,
            rounding_scheme=rounding_scheme,
        )
        results = qrao.solve()
        self.assertIsInstance(results.samples[0].fval, float)
        self.assertIsInstance(results.relaxed_results, MinimumEigensolverResult)
        self.assertIsInstance(results.rounding_results, RoundingResult)
        self.assertIsInstance(results.relaxed_fval, float)
        self.assertIsInstance(repr(results), str)

    def test_empty_encoding(self):
        """Test that an exception is raised if the encoding has no qubits"""
        estimator = Estimator(options={"shots": 100})
        vqe = VQE(
            ansatz=self.ansatz,
            optimizer=SPSA(maxiter=1, learning_rate=0.01, perturbation=0.1),
            estimator=estimator,
        )
        encoding = QuantumRandomAccessEncoding(3)
        with self.assertRaises(ValueError):
            QuantumRandomAccessOptimizer(encoding=encoding, min_eigen_solver=vqe)
