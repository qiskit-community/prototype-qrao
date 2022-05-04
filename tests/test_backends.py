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

"""Test QRAO steps on various hardware and simulator backends"""

import pytest

from docplex.mp.model import Model

from qiskit import Aer, BasicAer
from qiskit.utils import QuantumInstance
from qiskit.algorithms.minimum_eigen_solvers import VQE
from qiskit.circuit.library import RealAmplitudes
from qiskit.algorithms.optimizers import SPSA
from qiskit_optimization.algorithms import OptimizationResultStatus
from qiskit_optimization.translators import from_docplex_mp
from qiskit_ibm_provider import IBMProvider, least_busy, IBMAccountError

from qrao import (
    QuantumRandomAccessOptimizer,
    QuantumRandomAccessEncoding,
    MagicRounding,
)

# TODO:
# - update these tests to include solution checking once behavior can be made
# - deterministic.
#    - This might just require us to set seeds in the QuantumInstance and
#    - remove that as an argument altogether.

backends = [
    (BasicAer.get_backend, "qasm_simulator"),
    (Aer.get_backend, "qasm_simulator"),
    (Aer.get_backend, "statevector_simulator"),
    (Aer.get_backend, "aer_simulator"),
    (Aer.get_backend, "aer_simulator_statevector"),
    (Aer.get_backend, "aer_simulator_density_matrix"),
    (Aer.get_backend, "aer_simulator_matrix_product_state"),
    # The following takes forever, haven't yet waited long enough to know the
    # real timescale
    # (Aer.get_backend, "aer_simulator_extended_stabilizer"),
]


@pytest.fixture(scope="module")
def my_encoding():
    # Load small reference problem
    elist = [(0, 1), (0, 4), (0, 3), (1, 2), (1, 5), (2, 3), (2, 4), (4, 5), (5, 3)]
    num_nodes = 6
    mod = Model("maxcut")
    nodes = list(range(num_nodes))
    var = [mod.binary_var(name="x" + str(i)) for i in nodes]
    mod.maximize(mod.sum((var[i] + var[j] - 2 * var[i] * var[j]) for i, j in elist))
    problem = from_docplex_mp(mod)
    encoding = QuantumRandomAccessEncoding(max_vars_per_qubit=3)
    encoding.encode(problem)
    return encoding


@pytest.fixture(scope="module")
def my_ansatz(my_encoding):
    return RealAmplitudes(my_encoding.num_qubits)


@pytest.mark.parametrize("relaxed_backend", backends)
@pytest.mark.parametrize("rounding_backend", backends)
@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
@pytest.mark.filterwarnings(
    "ignore:.*statevector_simulator.*:UserWarning"
)  # ignore magic rounding's UserWarning when using statevector_simulator
@pytest.mark.backend
def test_backend(relaxed_backend, rounding_backend, my_encoding, my_ansatz, shots=3):
    def cb(f, *args):
        "Construct backend"
        return f(*args)

    relaxed_qi = QuantumInstance(backend=cb(*relaxed_backend), shots=shots)
    rounding_qi = QuantumInstance(backend=cb(*rounding_backend), shots=shots)
    vqe = VQE(
        ansatz=my_ansatz,
        optimizer=SPSA(maxiter=1, learning_rate=0.01, perturbation=0.1),
        quantum_instance=relaxed_qi,
    )
    rounding_scheme = MagicRounding(rounding_qi)
    qrao = QuantumRandomAccessOptimizer(
        encoding=my_encoding, min_eigen_solver=vqe, rounding_scheme=rounding_scheme
    )
    result = qrao.solve()
    assert result.status == OptimizationResultStatus.SUCCESS


@pytest.mark.backend
def test_magic_rounding_on_hardware_backend(my_encoding, my_ansatz):
    """Test *magic rounding* on a hardware backend, if available."""
    try:
        provider = IBMProvider()
    except IBMAccountError:
        pytest.skip("No hardware backend available")
    print(f"Encoding requires {my_encoding.num_qubits} qubits")
    backend = least_busy(
        provider.backends(
            filters=lambda x: x.configuration().n_qubits >= my_encoding.num_qubits,
            simulator=False,
        )
    )
    print(f"Using backend: {backend}")
    relaxed_qi = QuantumInstance(backend=Aer.get_backend("aer_simulator"), shots=100)
    rounding_qi = QuantumInstance(backend=backend, shots=32)
    vqe = VQE(
        ansatz=my_ansatz,
        optimizer=SPSA(maxiter=1, learning_rate=0.01, perturbation=0.1),
        quantum_instance=relaxed_qi,
    )
    rounding_scheme = MagicRounding(quantum_instance=rounding_qi)
    qrao = QuantumRandomAccessOptimizer(
        encoding=my_encoding, min_eigen_solver=vqe, rounding_scheme=rounding_scheme
    )
    result = qrao.solve()
    assert result.status == OptimizationResultStatus.SUCCESS
