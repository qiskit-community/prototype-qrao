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

"""Magic bases rounding"""

import numbers
import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.opflow import PrimitiveOp
from qiskit.providers import Backend
from qiskit.utils import QuantumInstance

from .encoding import QuantumRandomAccessEncoding, change_to_n1p_qrac_basis
from .rounding_common import (
    RoundingContext,
    RoundingResult,
    RoundingScheme,
    RoundingSolutionSample,
)

_invalid_backend_names = [
    "aer_simulator_unitary",
    "aer_simulator_superop",
    "unitary_simulator",
    "pulse_simulator",
]

# Prior to Qiskit Terra 0.20.1, the BasicAer and hardware backends fail if the
# shots count is provided as an np.int64.  So, we define this function, which
# will return an int if passed either, so that the int can then be passed as
# the shots count. Fixed in https://github.com/Qiskit/qiskit-terra/pull/7824
def _ensure_int(n: numbers.Integral) -> int:
    """Convert int-like quantity (e.g. ``numpy.int64``) to int"""
    return int(n)


def _backend_name(backend: Backend) -> str:
    """Return the backend name in a way that is agnostic to Backend version"""
    # See qiskit.utils.backend_utils in qiskit-terra for similar examples
    if backend.version <= 1:
        return backend.name()
    return backend.name


def _is_original_statevector_simulator(backend: Backend) -> bool:
    """Return True if the original statevector simulator"""
    return _backend_name(backend) == "statevector_simulator"


def _get_parity(n):
    parity = 0
    while n:
        parity = ~parity
        n = n & (n - 1)
    return parity >= 0


def _get_bitfield(n, length):
    return [n >> i & 1 for i in range(length - 1, -1, -1)]


class MagicRoundingResult(RoundingResult):
    """Result of magic rounding"""

    def __init__(
        self,
        samples: List[RoundingSolutionSample],
        *,
        bases=None,
        basis_shots=None,
        basis_counts=None,
        time_taken=None,
    ):
        self._bases = bases
        self._basis_shots = basis_shots
        self._basis_counts = basis_counts
        super().__init__(samples, time_taken=time_taken)

    @property
    def bases(self):
        return self._bases

    @property
    def basis_shots(self):
        return self._basis_shots

    @property
    def basis_counts(self):
        return self._basis_counts


class MagicRounding(RoundingScheme):
    """ "Magic rounding" method

    This method is described in https://arxiv.org/abs/2111.03167v2.

    """

    def __init__(
        self,
        quantum_instance: QuantumInstance,
        *,
        basis_sampling: str = "uniform",
        seed: Optional[int] = None,
    ):
        """
        Args:

            quantum_instance: Provides the ``Backend`` for quantum execution
                and the ``shots`` count (i.e., the number of samples to collect
                from the magic bases).

            basis_sampling: Method to use for sampling the magic bases.  Must
                be either ``"uniform"`` (default) or ``"weighted"``.
                ``"uniform"`` samples all magic bases uniformly, and is the
                method described in https://arxiv.org/abs/2111.03167v2.
                ``"weighted"`` attempts to choose bases strategically using the
                Pauli expectation values from the minimum eigensolver.
                However, the approximation bounds given in
                https://arxiv.org/abs/2111.03167v2 apply only to ``"uniform"``
                sampling.

            seed: Seed for random number generator, which is used to sample the
                magic bases.

        """
        if basis_sampling not in ("uniform", "weighted"):
            raise ValueError(
                f"'{basis_sampling}' is not an implemented sampling method. "
                "Please choose either 'uniform' or 'weighted'."
            )
        self.quantum_instance = quantum_instance
        self.rng = np.random.RandomState(seed)
        self._basis_sampling = basis_sampling
        super().__init__()

    @property
    def shots(self) -> int:
        """Shots count as configured by the given ``quantum_instance``."""
        return self.quantum_instance.run_config.shots

    @property
    def basis_sampling(self):
        """Basis sampling method (either ``"uniform"`` or ``"weighted"``)."""
        return self._basis_sampling

    @property
    def quantum_instance(self) -> QuantumInstance:
        """Provides the ``Backend`` and the ``shots`` (samples) count."""
        return self._quantum_instance

    @quantum_instance.setter
    def quantum_instance(self, quantum_instance: QuantumInstance) -> None:
        backend_name = _backend_name(quantum_instance.backend)
        if backend_name in _invalid_backend_names:
            raise ValueError(f"{backend_name} is not supported.")
        if _is_original_statevector_simulator(quantum_instance.backend):
            warnings.warn(
                'Use of "statevector_simulator" is discouraged because it effectively '
                "brute-forces all possible solutions.  We suggest using the newer "
                '"aer_simulator_statevector" instead.'
            )
        self._quantum_instance = quantum_instance

    @staticmethod
    def _get_dvar_values_from_bit(num_dvars: int, basis: int, bit: int):
        """
        Return the decision variables values measured from the qubit encoding them
        in the given basis.
        """
        dvars = [dvars for dvars in range(2 ** (num_dvars)) if _get_parity(dvars)][
            basis
        ]
        if bit:
            dvars = ~dvars
        dvars = _get_bitfield(dvars, num_dvars)
        return dvars

    def _get_dvar_values_from_bits(
        self,
        bits: List[int],
        bases: List[int],
        dvar_to_op: Dict[int, Tuple[int, PrimitiveOp]],
        qubit_to_dvars: List[List[int]],
    ) -> List[int]:
        """
        Return the decision variables values measured from the qubits encoding them
        in the given bases.
        """
        dvar_values = {}
        for qubit, dvars in enumerate(qubit_to_dvars):
            qubit_dvar_values = self._get_dvar_values_from_bit(
                len(dvars), bases[qubit], bits[qubit]
            )
            for dvar in dvars:
                _, operator = dvar_to_op[dvar]
                dvar_values[dvar] = qubit_dvar_values[
                    QuantumRandomAccessEncoding.NUM_DVARS_TO_OPS[len(dvars)].index(
                        operator
                    )
                ]
        return [dvar_values[dvar] for dvar in range(len(dvar_values))]

    def _get_basis_measurement_circuits(
        self,
        circuit: QuantumCircuit,
        bases: List[List[int]],
        measure: bool,
        qubit_to_dvars,
    ) -> List[QuantumCircuit]:
        """
        Return the circuits measuring the qubits encoding the decision variables
        in the given bases.
        """
        measured_circuits = []
        for basis in bases:
            measured_circuit = circuit.copy()
            for (qubit, dvars), operator in zip(enumerate(qubit_to_dvars), basis):
                measured_circuit.append(
                    change_to_n1p_qrac_basis(len(dvars), operator).inverse(),
                    qargs=[qubit],
                )
            if measure:
                measured_circuit.measure_all()
            measured_circuits.append(measured_circuit)
        return measured_circuits

    def _evaluate_magic_bases(self, circuit, bases, basis_shots, qubit_to_dvars):
        """
        Given a circuit you wish to measure, a list of magic bases to measure,
        and a list of the shots to use for each magic basis configuration,
        measure the provided circuit in the magic bases given and return the counts
        dictionaries associated with each basis measurement.

        len(bases) == len(basis_shots) == len(basis_counts)
        """
        measure = not _is_original_statevector_simulator(self.quantum_instance.backend)
        circuits = self._get_basis_measurement_circuits(
            circuit, bases, measure, qubit_to_dvars
        )

        # Execute each of the rotated circuits and collect the results.

        # Batch the circuits into jobs where each group has the same number of
        # shots, so that you can wait for the queue as few times as possible if
        # using hardware.
        circuit_indices_by_shots: Dict[int, List[int]] = {}
        assert len(circuits) == len(basis_shots)
        for i, shots in enumerate(basis_shots):
            circuit_indices_by_shots.setdefault(_ensure_int(shots), []).append(i)

        basis_counts: List[Optional[Dict[str, int]]] = [None] * len(circuits)
        overall_shots = self.quantum_instance.run_config.shots
        try:
            for shots, indices in sorted(
                circuit_indices_by_shots.items(), reverse=True
            ):
                self.quantum_instance.set_config(shots=shots)
                result = self.quantum_instance.execute([circuits[i] for i in indices])
                counts_list = result.get_counts()
                if not isinstance(counts_list, List):
                    # This is the only case where this should happen, and that
                    # it does at all (namely, when a single-element circuit
                    # list is provided) is a weird API quirk of Qiskit.
                    assert len(indices) == 1
                    counts_list = [counts_list]
                assert len(indices) == len(counts_list)
                for i, counts in zip(indices, counts_list):
                    basis_counts[i] = counts
        finally:
            # We've temporarily modified quantum_instance; now we restore it to
            # its initial state.
            self.quantum_instance.set_config(shots=overall_shots)
        assert None not in basis_counts

        # Process the outcomes and extract expectation of decision variables.

        # The "statevector_simulator", unlike all the others, returns
        # probabilities instead of integer counts.  So if probabilities are
        # detected, we rescale them.
        if any(
            any(not isinstance(x, numbers.Integral) for x in counts.values())
            for counts in basis_counts
        ):
            basis_counts = [
                {key: val * basis_shots[i] for key, val in counts.items()}
                for i, counts in enumerate(basis_counts)
            ]

        return basis_counts

    def _get_dvar_values_counts(self, basis_counts, bases, dvar_to_op, qubit_to_dvars):
        """
        Given a list of bases, basis_shots, and basis_counts, convert
        each observed bitstrings to its corresponding decision variable
        configuration. Return the counts of each decision variable configuration.
        """
        dvar_values_counts = {}
        for i, counts in enumerate(basis_counts):
            basis = bases[i]
            # For each measurement outcome...
            for bits, count in counts.items():
                # For each bit in the observed bitstring...
                solution = self._get_dvar_values_from_bits(
                    list(map(int, list(bits))), basis, dvar_to_op, qubit_to_dvars
                )
                solution = "".join([str(int(bit)) for bit in solution])
                if solution in dvar_values_counts:
                    dvar_values_counts[solution] += count
                else:
                    dvar_values_counts[solution] = count
        return dvar_values_counts

    def _sample_bases_uniform(self, qubit_to_dvars):
        bases = [
            [self.rng.choice(2 ** (len(variables) - 1)) for variables in qubit_to_dvars]
            for _ in range(self.shots)
        ]
        bases, basis_shots = np.unique(bases, axis=0, return_counts=True)
        return bases, basis_shots

    def _sample_bases_weighted(self, qubit_to_dvars, trace_values):
        """Perform weighted sampling from the expectation values.

        The goal is to make smarter choices about which bases to measure in
        using the trace values.
        """
        trace_values = np.clip(trace_values, -1, 1)
        basis_probabilities = []
        for variables in qubit_to_dvars:
            operators = [0.5 * (1 - trace_values[variable]) for variable in variables]
            basis_operator_probabilities = []
            for basis in range(2 ** (len(variables) - 1)):
                operator_signs = map(int, format(basis, f"0{len(variables)}b"))
                positive_product_factors = []
                negative_product_factors = []
                for variable, is_negative in enumerate(operator_signs):
                    if is_negative:
                        negative_product_factors.append(operators[variable])
                        positive_product_factors.append(1 - operators[variable])
                    else:
                        positive_product_factors.append(operators[variable])
                        negative_product_factors.append(1 - operators[variable])
                positive_product = np.prod(positive_product_factors)
                negative_product = np.prod(negative_product_factors)
                basis_operator_probabilities.append(
                    np.real(positive_product + negative_product)
                )
            basis_probabilities.append(basis_operator_probabilities)
        bases = [
            [
                self.rng.choice(len(probabilities), p=probabilities)
                for probabilities in basis_probabilities
            ]
            for _ in range(self.shots)
        ]
        bases, basis_shots = np.unique(bases, axis=0, return_counts=True)
        return bases, basis_shots

    def round(self, ctx: RoundingContext) -> MagicRoundingResult:
        """Perform magic rounding"""

        start_time = time.time()
        trace_values = ctx.trace_values
        circuit = ctx.circuit

        if circuit is None:
            raise NotImplementedError(
                "Magic rounding requires a circuit to be available.  Perhaps try "
                "semideterministic rounding instead."
            )

        # We've already checked that it is one of these two in the constructor.
        if self.basis_sampling == "uniform":
            bases, basis_shots = self._sample_bases_uniform(ctx.qubit_to_dvars)
        elif self.basis_sampling == "weighted":
            if trace_values is None:
                raise NotImplementedError(
                    "Magic rounding with weighted sampling requires the trace values "
                    "to be available, but they are not."
                )
            bases, basis_shots = self._sample_bases_weighted(
                ctx.qubit_to_dvars, trace_values
            )
        else:  # pragma: no cover
            raise NotImplementedError(
                f'No such basis sampling method: "{self.basis_sampling}".'
            )

        assert self.shots == np.sum(basis_shots)

        # For each of the magic bases sampled above, measure
        # basis_shots times and return the circuit results.
        basis_counts = self._evaluate_magic_bases(
            circuit, bases, basis_shots, ctx.qubit_to_dvars
        )

        # Keys are decision variable values.
        # Values are the total number of observations.
        solution_counts = self._get_dvar_values_counts(
            basis_counts, bases, ctx.dvar_to_op, ctx.qubit_to_dvars
        )

        solution_samples = [
            RoundingSolutionSample(
                x=np.asarray([int(bit) for bit in solution]),
                probability=count / self.shots,
            )
            for solution, count in solution_counts.items()
        ]

        assert np.isclose(
            sum(solution_counts.values()), self.shots
        ), f"{sum(solution_counts.values())} != {self.shots}"
        assert len(bases) == len(basis_shots) == len(basis_counts)
        stop_time = time.time()

        return MagicRoundingResult(
            samples=solution_samples,
            bases=bases,
            basis_shots=basis_shots,
            basis_counts=basis_counts,
            time_taken=stop_time - start_time,
        )
