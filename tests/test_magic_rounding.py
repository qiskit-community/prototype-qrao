# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for MagicRounding."""

import unittest
import pytest

import numpy as np
from docplex.mp.model import Model

from qiskit import Aer, QuantumCircuit
from qiskit.utils import QuantumInstance
from qiskit.opflow import (
    StateFn,
    X,
    Y,
    Z,
)

from qiskit_optimization.translators import from_docplex_mp

from qrao import (
    RoundingContext,
    MagicRounding,
)
from qrao.encoding import state_from_dvar_values, q2vars_from_var2op

# pylint: disable=protected-access


class TestMagicRounding(unittest.TestCase):
    """Test MagicRounding Class"""

    def setUp(self):
        # load problem, define encoding etc
        # instantiate MagicRounding
        # things here don't change (often)
        super().setUp()
        self.gate_circ = QuantumCircuit(2)
        self.gate_circ.h(0)
        self.gate_circ.h(1)
        self.gate_circ.z(1)
        self.gate_circ.cx(0, 1)
        self.gate_circ.s(0)
        self.gate_circ.save_statevector()

        self.deterministic_trace_vals = [
            [1, 1, 1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, -1, -1],
        ]

        elist = [(0, 1), (0, 4), (0, 3), (1, 2), (1, 5), (2, 3), (2, 4), (4, 5), (5, 3)]
        num_nodes = 6
        mod = Model("maxcut")
        nodes = list(range(num_nodes))
        var = [mod.binary_var(name="x" + str(i)) for i in nodes]
        mod.maximize(mod.sum((var[i] + var[j] - 2 * var[i] * var[j]) for i, j in elist))
        self.problem = from_docplex_mp(mod)
        self.rounding_qi = QuantumInstance(
            backend=Aer.get_backend("aer_simulator"), shots=10
        )

    # Start test cases in order of increasing complexity
    # toy problems
    # harder problems w/ minimal inputs
    # harder problems w/ more inputs
    # negative test cases
    def test_round_on_gate_and_sv_circs(self):
        """Test MagicRounding"""
        ops = [X, Y, Z]
        var2op = {i: (i // 3, ops[i % 3]) for i in range(3)}
        qi = QuantumInstance(backend=Aer.get_backend("aer_simulator"), shots=1000)
        magic = MagicRounding(
            quantum_instance=qi,
            basis_sampling="weighted",
        )
        # subtest for gate based and sv based etc
        with self.subTest("Gate Based Magic Uniform Rounding"):
            for m0 in range(2):
                for m1 in range(2):
                    for m2 in range(2):
                        qrac_gate_circ = state_from_dvar_values(m0, m1, m2).to_circuit()
                        magic_basis = 2 * (m1 ^ m2) + (m0 ^ m2)
                        tv = self.deterministic_trace_vals[magic_basis]
                        rounding_context = RoundingContext(
                            var2op, trace_values=tv, circuit=qrac_gate_circ
                        )
                        rounding_res = magic.round(rounding_context)
                        self.assertEqual(
                            rounding_res.samples[0].x.tolist(), [m0, m1, m2]
                        )
                        self.assertEqual(rounding_res.samples[0].probability, 1)

        with self.subTest("SV Based Magic Uniform Rounding"):
            for m0 in range(2):
                for m1 in range(2):
                    for m2 in range(2):
                        qrac_gate_circ = state_from_dvar_values(m0, m1, m2).to_circuit()
                        sv = StateFn(qrac_gate_circ).eval().primitive
                        qrac_sv_circ = QuantumCircuit(1)
                        qrac_sv_circ.initialize(sv)
                        magic_basis = 2 * (m1 ^ m2) + (m0 ^ m2)
                        tv = self.deterministic_trace_vals[magic_basis]
                        rounding_context = RoundingContext(
                            var2op, trace_values=tv, circuit=qrac_sv_circ
                        )
                        rounding_res = magic.round(rounding_context)
                        self.assertEqual(
                            rounding_res.samples[0].x.tolist(), [m0, m1, m2]
                        )
                        self.assertEqual(rounding_res.samples[0].probability, 1)

    def test_evaluate_magic_bases(self):
        qi = QuantumInstance(backend=Aer.get_backend("aer_simulator"), shots=1000)
        magic = MagicRounding(
            quantum_instance=qi,
            basis_sampling="weighted",
        )
        for m0 in range(2):
            for m1 in range(2):
                for m2 in range(2):
                    qrac_state = state_from_dvar_values(m0, m1, m2).to_circuit()
                    bases = [[2 * (m1 ^ m2) + (m0 ^ m2)]]
                    basis_counts = magic._evaluate_magic_bases(
                        qrac_state,
                        bases=bases,
                        basis_shots=[10],
                        q2vars=[[0, 1, 2]],
                    )
                    self.assertEqual(len(basis_counts), 1)
                    self.assertEqual(int(list(basis_counts[0].keys())[0]), m0 ^ m1 ^ m2)

    def test_dv_counts(self):
        """
        Checks that the dv_counts method unpacks these measurement outcomes
        properly. This also effectively tests `unpack_measurement_outcome`.
        """
        ops = [X, Y, Z]
        var2op = {i: (i // 3, ops[i % 3]) for i in range(6)}
        magic = MagicRounding(self.rounding_qi)
        compute_dv_counts = magic._compute_dv_counts
        solns = []
        for b0 in range(4):
            for b1 in range(4):
                for outcome in range(4):
                    bases = [[b0, b1]]
                    basis_counts = [{"{:02b}".format(outcome): 1}]
                    dv_counts = compute_dv_counts(
                        basis_counts, bases, var2op, q2vars_from_var2op(var2op)
                    )
                    solns.append(list(dv_counts.keys())[0])
        ref = [
            "000000",
            "000111",
            "111000",
            "111111",
            "000011",
            "000100",
            "111011",
            "111100",
            "000101",
            "000010",
            "111101",
            "111010",
            "000110",
            "000001",
            "111110",
            "111001",
            "011000",
            "011111",
            "100000",
            "100111",
            "011011",
            "011100",
            "100011",
            "100100",
            "011101",
            "011010",
            "100101",
            "100010",
            "011110",
            "011001",
            "100110",
            "100001",
            "101000",
            "101111",
            "010000",
            "010111",
            "101011",
            "101100",
            "010011",
            "010100",
            "101101",
            "101010",
            "010101",
            "010010",
            "101110",
            "101001",
            "010110",
            "010001",
            "110000",
            "110111",
            "001000",
            "001111",
            "110011",
            "110100",
            "001011",
            "001100",
            "110101",
            "110010",
            "001101",
            "001010",
            "110110",
            "110001",
            "001110",
            "001001",
        ]
        self.assertTrue(np.all(np.array(ref) == np.array(solns)))

    def test_sample_bases_weighted(self):
        """
        There are a few settings of the trace values which
        cause the magic basis sampling probabilities to be deterministic.
        I pass these through (for a 2 qubit, 6 var example) and verify
        that the outputs are correctly shaped and are deterministic.

        Note that these input trace values are non-physical
        """
        shots = 10
        num_nodes = 6
        num_qubits = 2
        rounding_qi = QuantumInstance(
            backend=Aer.get_backend("aer_simulator"), shots=shots
        )
        ops = [X, Y, Z]
        var2op = {i: (i // 3, ops[i % 3]) for i in range(num_nodes)}
        q2vars = q2vars_from_var2op(var2op)
        magic = MagicRounding(quantum_instance=rounding_qi, basis_sampling="weighted")
        sample_bases_weighted = magic._sample_bases_weighted

        stable_inputs = [
            ([1, 1, 1], 0),
            ([1, 1, -1], 1),
            ([1, -1, 1], 2),
            ([1, -1, -1], 3),
        ]

        for tv0, b0 in stable_inputs:
            for tv1, b1 in stable_inputs:
                tv = tv0 + tv1
                bases, basis_shots = sample_bases_weighted(q2vars, tv)
                self.assertTrue(np.all(np.array([b0, b1]) == bases))
                self.assertEqual(basis_shots, (shots,))
                self.assertEqual(bases.shape, (1, num_qubits))  # 1 == deterministic

        # Both trace values and a circuit must be provided
        with self.assertRaises(NotImplementedError):
            magic.round(RoundingContext(var2op, trace_values=[1.0]))
        with self.assertRaises(NotImplementedError):
            magic.round(RoundingContext(var2op, circuit=self.gate_circ))

    def test_sample_bases_uniform(self):
        """
        Verify that the outputs of uniform sampling are correctly shaped.
        """
        num_nodes = 3
        num_qubits = 1
        ops = [X, Y, Z]
        var2op = {i: (i // 3, ops[i % 3]) for i in range(num_nodes)}
        q2vars = q2vars_from_var2op(var2op)
        shots = 1000  # set high enough to "always" have four distinct results
        qi = QuantumInstance(backend=Aer.get_backend("aer_simulator"), shots=shots)
        magic = MagicRounding(
            quantum_instance=qi,
            basis_sampling="uniform",
        )
        bases, basis_shots = magic._sample_bases_uniform(q2vars)
        self.assertEqual(basis_shots.shape, (4,))
        self.assertEqual(np.sum(basis_shots), shots)
        self.assertEqual(bases.shape, (4, num_qubits))

        # A circuit must be provided, but trace values need not be
        circuit = QuantumCircuit(1)
        circuit.h(0)
        magic.round(RoundingContext(var2op, circuit=circuit))
        with self.assertRaises(NotImplementedError):
            magic.round(RoundingContext(var2op))


# def test_unsupported_qrac():
#     qi = QuantumInstance(backend=Aer.get_backend("aer_simulator"), shots=1000)
#     encoding = QuantumRandomAccessEncoding(2)
#     rounding = MagicRounding(quantum_instance=qi)
#     ctx = RoundingContext(encoding=encoding)
#     with pytest.raises(ValueError):
#         rounding.round(ctx)


def test_unsupported_backend():
    qi = QuantumInstance(Aer.get_backend("aer_simulator_unitary"), shots=100)
    with pytest.raises(ValueError):
        MagicRounding(quantum_instance=qi)


def test_unsupported_basis_sampling_method():
    qi = QuantumInstance(Aer.get_backend("aer_simulator"), shots=100)
    with pytest.raises(ValueError):
        MagicRounding(quantum_instance=qi, basis_sampling="foo")


@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
def test_magic_rounding_statevector_simulator():
    """Test magic rounding on the statevector simulator

    ... which behaves unlike the others, as the "counts" are probabilities, not
    integers, and so special care is required.
    """
    qi = QuantumInstance(Aer.get_backend("statevector_simulator"), shots=10)
    ops = [X, Y, Z]
    var2op = {i: (i // 3, ops[i % 3]) for i in range(3)}
    with pytest.warns(UserWarning):
        magic = MagicRounding(
            quantum_instance=qi,
            basis_sampling="weighted",
        )
    circ = QuantumCircuit(2)
    circ.h(0)
    circ.h(1)
    circ.cx(0, 1)
    ctx = RoundingContext(var2op, circuit=circ, trace_values=[1, 1, 1])
    res = magic.round(ctx)
    assert sum(s.probability for s in res.samples) == pytest.approx(1)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
