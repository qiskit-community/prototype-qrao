# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Quantum Random Access Encoding module.

Contains code dealing with QRACs (quantum random access codes) and preparation
of such states.

.. autosummary::
   :toctree: ../stubs/

   change_to_n1p_qrac_basis
   get_dvars_encoding_state
   get_problem_encoding_state
   QuantumRandomAccessEncoding

"""

from functools import reduce
from itertools import chain
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import retworkx as rx
from qiskit import QuantumCircuit
from qiskit.opflow import (
    CircuitOp,
    CircuitStateFn,
    I,
    One,
    PauliOp,
    PauliSumOp,
    PrimitiveOp,
    StateFn,
    X,
    Y,
    Z,
    Zero,
)
from qiskit_optimization.problems.quadratic_program import QuadraticProgram


def _ceildiv(n: int, d: int) -> int:
    """Perform ceiling division in integer arithmetic

    >>> _ceildiv(0, 3)
    0
    >>> _ceildiv(1, 3)
    1
    >>> _ceildiv(3, 3)
    1
    >>> _ceildiv(4, 3)
    2
    """
    return (n - 1) // d + 1


def change_to_n1p_qrac_basis(num_dvars, basis) -> QuantumCircuit:
    """Return the basis change corresponding to the (num_dvars, 1, p)-QRAC
       for the given basis

    Args:
        num_dvars: The number of decision variables encoded in the qubit.
        state: The index of the (num_dvars, 1, p)-QRAC basis to change to.

    Returns:
        The ``QuantumCircuit`` implementing the change of basis.

    """

    if num_dvars not in (1, 2, 3):
        raise ValueError(f"num_dvars must be 1, 2, or 3, not {num_dvars}.")
    n_states = 2 ** (num_dvars - 1)
    if basis not in range(0, n_states):
        raise ValueError(f"state must be in [0, {n_states}], not {basis}.")

    beta = np.arccos(1 / np.sqrt(num_dvars))

    basis_change_qc = QuantumCircuit(1)

    # fmt: off
    if basis == 0:
        basis_change_qc.r(0     + -1 * beta, -1 * np.pi / n_states, 0)
    elif basis == 1:
        basis_change_qc.r(np.pi + -1 * beta,  1 * np.pi / n_states, 0)
    elif basis == 2:
        basis_change_qc.r(np.pi +  1 * beta,  1 * np.pi / n_states, 0)
    elif basis == 3:
        basis_change_qc.r(0     +  1 * beta, -1 * np.pi / n_states, 0)
    # fmt: on

    return basis_change_qc


def get_dvars_encoding_state(*dvar_values: int) -> CircuitStateFn:
    """Prepare a single-qubit QRAC state from a list of decision variable values.

        This function accepts 1, 2, or 3 decision variables, in which case it
        generates a 1-QRAC, 2-QRAC, or 3-QRAC, respectively.

    Args:
        dvar_values: The values of the decision variables to encode. Each decision
                     variable must have value 0 or 1.

    Returns:
        The single-qubit QRAC circuit state function.

    """
    num_dvars = len(dvar_values)

    if num_dvars not in (1, 2, 3):
        raise TypeError(
            f"state_from_dvars can take up to 3 decision variables, not {num_dvars}."
        )
    if not all(dvar_value in (0, 1) for dvar_value in dvar_values):
        raise ValueError("Each decision variable must have value 0 or 1.")

    has_even_parity = sum(dvar_values) % 2
    has_even_count = num_dvars % 2

    basis = sum(
        [
            (2**i) * (dvar_values[i] ^ dvar_values[num_dvars - 1])
            for i in range(num_dvars - 1)
        ]
    )
    state = One if (has_even_parity if has_even_count else dvar_values[0]) else Zero
    return (
        CircuitOp(change_to_n1p_qrac_basis(num_dvars, basis)) @ state
    ).to_circuit_op()


def get_problem_encoding_state(
    dvar_values: Union[Dict[int, int], List[int]],
    qubit_to_dvars: List[List[int]],
    max_dvars_per_qubit: int,
) -> CircuitStateFn:
    """Prepare a composite state of single-qubit QRAC states encoding the specified
       problem from a list of decision variable values, assignments of decision
       variables to qubits, and the maximum number of decision variables that can be
       encoded on a given qubit.

    Args:
        dvar_values: The values of the decision variables to encode. Each decision
                     variable must have value 0 or 1.

    Returns:
        The composite circuit state function encoding the specified problem.

    """
    remaining_dvars = set(
        dvar_values if isinstance(dvar_values, dict) else range(len(dvar_values))
    )
    qubits_dvar_values = []
    for qubit_dvars in qubit_to_dvars:
        if len(qubit_dvars) < 1:
            raise ValueError(
                "Each qubit must have at least one decision variable assigned to it."
            )
        if len(qubit_dvars) > max_dvars_per_qubit:
            raise ValueError(
                "Each qubit is expected to be associated with at most "
                f"`max_dvars_per_qubit` ({max_dvars_per_qubit}) variables, "
                f"not {len(qubit_dvars)} variables."
            )
        qubit_dvar_values: List[int] = []
        for dvar in qubit_dvars:
            try:
                qubit_dvar_values.append(dvar_values[dvar])
            except (KeyError, IndexError):
                raise ValueError(
                    f"Decision variable not included in dvars: {dvar}"
                ) from None
            try:
                remaining_dvars.remove(dvar)
            except KeyError:
                raise ValueError(
                    f"Unused decision variable(s) in dvars: {remaining_dvars}"
                ) from None
        qubits_dvar_values.append(qubit_dvar_values)
    if remaining_dvars:
        raise ValueError(
            f"Not all dvars were included in qubit_to_dvars: {remaining_dvars}"
        )
    dvars_encoding_states = [
        get_dvars_encoding_state(*qubit_dvar_values)
        for qubit_dvar_values in qubits_dvar_values
    ]
    problem_encoding_state = reduce(lambda x, y: x ^ y, dvars_encoding_states)
    return problem_encoding_state


def get_qubit_from_op_assignment(
    dvar_to_op: Dict[int, Tuple[int, PrimitiveOp]]
) -> List[List[int]]:
    """Get qubit assignments from op assignment for decision variables."""
    num_qubits = max(qubit_index for qubit_index, _ in dvar_to_op.values()) + 1
    qubit_to_dvars: List[List[int]] = [[] for i in range(num_qubits)]
    for dvar, (qubit, _) in dvar_to_op.items():
        qubit_to_dvars[qubit].append(dvar)
    return qubit_to_dvars


class QuantumRandomAccessEncoding:
    """This class specifies a Quantum Random Access Code that can be used to encode
    the binary variables of a QUBO (quadratic unconstrained binary optimization
    problem).

    Args:
        max_dvars_per_qubit: maximum possible compression ratio.
            Supported values are 1, 2, or 3.

    """

    # This defines the convention of the Pauli operators (and their ordering)
    # for each encoding scheme.
    NUM_DVARS_TO_OPS = {
        1: [Z],  # (1,1,1) QRAC
        2: [X, Z],  # (2,1,p) QRAC, p ≈ 0.85
        3: [X, Y, Z],  # (3,1,p) QRAC, p ≈ 0.79
    }

    def __init__(self, max_dvars_per_qubit: int = 3):
        if max_dvars_per_qubit not in (1, 2, 3):
            raise ValueError("max_dvars_per_qubit must be 1, 2, or 3")
        self._max_dvars_per_qubit = max_dvars_per_qubit

        self._problem: Optional[QuadraticProgram] = None
        self._offset: Optional[float] = None

        self._dvar_to_op: Dict[int, Tuple[int, PrimitiveOp]] = {}
        self._qubit_to_dvars: List[List[int]] = []
        self._qubit_op: Optional[Union[PauliOp, PauliSumOp]] = None

        self._frozen = False

    @property
    def num_qubits(self) -> int:
        return len(self._qubit_to_dvars)

    @property
    def num_dvars(self) -> int:
        return len(self._dvar_to_op)

    @property
    def max_dvars_per_qubit(self) -> int:
        return self._max_dvars_per_qubit

    @property
    def dvar_to_op(self) -> Dict[int, Tuple[int, PrimitiveOp]]:
        return self._dvar_to_op

    @property
    def qubit_to_dvars(self) -> List[List[int]]:
        return self._qubit_to_dvars

    @property
    def compression_ratio(self) -> float:
        return self.num_dvars / self.num_qubits

    @property
    def minimum_recovery_probability(self) -> float:
        return (1 + 1 / np.sqrt(self.max_dvars_per_qubit)) / 2

    @property
    def qubit_op(self) -> Union[PauliOp, PauliSumOp]:
        if self._qubit_op is None:
            raise AttributeError(
                "No objective function has been provided from which a "
                "qubit Hamiltonian can be constructed. Please use the "
                "encode method if you wish to manually compile "
                "this field."
            )
        return self._qubit_op

    @property
    def offset(self) -> float:
        if self._offset is None:
            raise AttributeError(
                "No objective function has been provided from which a "
                "qubit Hamiltonian can be constructed. Please use the "
                "encode method if you wish to manually compile "
                "this field."
            )
        return self._offset

    @property
    def problem(self) -> QuadraticProgram:
        if self._problem is None:
            raise AttributeError(
                "No quadratic program has been associated with this object. "
                "Please use the encode method if you wish to do so."
            )
        return self._problem

    def _add_dvars(self, dvars: List[int]) -> None:

        # NOTE: If this is called multiple times, it *always* adds an
        # additional qubit (see final line), even if aggregating them into a
        # single call would have resulted in fewer qubits.

        self.ensure_thawed()
        if self._qubit_op is not None:
            raise RuntimeError(
                "_add_dvars() cannot be called once terms have been added "
                "to the operator, as the number of qubits must thereafter "
                "remain fixed."
            )
        if not dvars:
            return
        if len(dvars) != len(set(dvars)):
            raise ValueError("Added decision variables must be unique")
        for dvar in dvars:
            if dvar in self.dvar_to_op:
                raise ValueError(
                    "Added decision variables cannot collide with existing ones"
                )

        old_num_qubits = len(self.qubit_to_dvars)
        num_new_qubits = _ceildiv(len(dvars), self.max_dvars_per_qubit)

        # Assign each decision variable a qubit.
        for _ in range(num_new_qubits):
            self.qubit_to_dvars.append([])
        for i, dvar in enumerate(dvars):
            qubit, _ = divmod(i, self.max_dvars_per_qubit)
            qubit_index = old_num_qubits + qubit
            self.qubit_to_dvars[qubit_index].append(dvar)

        # Assign each decision variable an operator.
        for i, dvar in enumerate(dvars):
            qubit, op = divmod(i, self.max_dvars_per_qubit)
            qubit_index = old_num_qubits + qubit
            assert dvar not in self.dvar_to_op
            num_dvars_in_qubit = len(self.qubit_to_dvars[qubit_index])
            self.dvar_to_op[dvar] = (
                qubit_index,
                self.NUM_DVARS_TO_OPS[num_dvars_in_qubit][op],
            )

    def _add_term(self, w: float, *dvars: int) -> None:
        self.ensure_thawed()

        # Eq. (31) in https://arxiv.org/abs/2111.03167v2 assumes a weight-2
        # Pauli operator.  To generalize, we replace the `d` in that equation
        # with `d_prime`, defined as follows:
        d_prime = np.sqrt(
            np.prod([len(self.qubit_to_dvars[self.dvar_to_op[x][0]]) for x in dvars])
        )

        op = w * d_prime * self.term_to_op(*dvars)

        # We perform the following short-circuit *after* calling term_to_op so at
        # least we have confirmed that the user provided a valid variables list.
        if w == 0.0:
            return
        if self._qubit_op is None:
            self._qubit_op = op
        else:
            self._qubit_op += op

    def term_to_op(self, *dvars: int) -> PauliOp:
        ops = [I] * self.num_qubits
        done_qubits = set()
        for dvar in dvars:
            qubit, op = self._dvar_to_op[dvar]
            if qubit in done_qubits:
                raise RuntimeError(f"Collision of variables: {dvars}.")
            ops[qubit] = op
            done_qubits.add(qubit)
        return reduce(lambda x, y: x ^ y, ops)

    @staticmethod
    def _find_variable_partition(quad: np.ndarray) -> Dict[int, List[int]]:
        num_nodes = quad.shape[0]
        assert quad.shape == (num_nodes, num_nodes)

        graph = rx.PyGraph()
        graph.add_nodes_from(range(num_nodes))
        graph.add_edges_from_no_data(list(zip(*np.where(quad != 0))))

        node_to_color = rx.graph_greedy_color(graph)
        color_to_node: Dict[int, List[int]] = {}
        for node, color in sorted(node_to_color.items()):
            color_to_node.setdefault(color, []).append(node)

        return color_to_node

    def encode(self, problem: QuadraticProgram) -> None:
        """Encode the (n,1,p) QRAC relaxed Hamiltonian of this problem.

            We associate to each binary decision variable one bit of a
            (n,1,p) Quantum Random Access Code. This is done in such a way that the
            given problem's objective function commutes with the encoding.

            After being called, the object will have the following attributes:
            qubit_op: The qubit operator encoding the input QuadraticProgram.
            offset: The constant value in the encoded Hamiltonian.
            problem: The ``problem`` used for encoding.

        Inputs:
            problem: A QuadraticProgram object encoding a QUBO optimization problem

        Raises:
            RuntimeError: if the ``problem`` isn't a QUBO or if the current
                object has been used already

        """

        # Ensure the encoding instance is fresh.
        if self.num_qubits > 0:
            raise RuntimeError(
                "Must call encode() on an Encoding that has not been used already"
            )
        # If the given problem has variables that are not binary, raise an error.
        if problem.get_num_vars() > problem.get_num_binary_vars():
            raise RuntimeError(
                "The type of all variables must be binary. "
                "You can use `QuadraticProgramToQubo` converter "
                "to convert integer variables to binary variables. "
                "If the problem contains continuous variables, `qrao` "
                "cannot handle it."
            )
        # If constraints exist on the given problem, raise an error.
        if problem.linear_constraints or problem.quadratic_constraints:
            raise RuntimeError(
                "There must be no constraint in the problem. "
                "You can use `QuadraticProgramToQubo` converter to convert "
                "constraints to penalty terms of the objective function."
            )

        self._problem = problem
        num_dvars = problem.get_num_vars()

        # Extract a sign corresponding to the kind of optimization problem:
        #   1 represents a minimization problem.
        #  -1 represents a maximization problem.
        sense = problem.objective.sense.value

        # Extract the constant term from the problem objective function.
        offset = problem.objective.constant * sense

        # Extract the linear terms from the problem objective function.
        linear_terms = np.zeros(num_dvars)
        for index, coefficient in problem.objective.linear.to_dict().items():
            weight = coefficient * sense / 2
            linear_terms[index] -= weight
            offset += weight

        # Extract the quadratic terms from the problem objective function.
        quadratic_terms = np.zeros((num_dvars, num_dvars))
        for (i, j), coefficient in problem.objective.quadratic.to_dict().items():
            weight = coefficient * sense / 4
            if i == j:
                linear_terms[i] -= 2 * weight
                offset += 2 * weight
            else:
                quadratic_terms[i, j] += weight
                linear_terms[i] -= weight
                linear_terms[j] -= weight
                offset += weight

        self._offset = offset

        # Find a partition of decision variables (a graph coloring is sufficient).
        dvars_partition = self._find_variable_partition(quadratic_terms)

        # The other methods of the current class allow for the variables to
        # have arbitrary integer indices [i.e., they need not correspond to
        # range(num_vars)], and the tests corresponding to this file ensure
        # that this works.  However, the current method is a high-level one
        # that takes a QuadraticProgram, which always has its variables
        # numbered sequentially.  Furthermore, other portions of the QRAO code
        # base [most notably the assignment of variable_ops in solve_relaxed()
        # and the corresponding result objects] assume that the variables are
        # numbered from 0 to (num_vars - 1).  So we enforce that assumption
        # here, both as a way of documenting it and to make sure
        # _find_variable_partition() returns a sensible result (in case the
        # user overrides it).
        assert sorted(chain.from_iterable(dvars_partition.values())) == list(
            range(num_dvars)
        )

        # Assign each decision variable a qubit and an operator in the circuit
        # encoding the problem.
        for _, dvar in sorted(dvars_partition.items()):
            self._add_dvars(sorted(dvar))

        # Add terms to a Hamiltonian encoding the problem from the decision
        # variable partition and the constant, linear, and qudratic terms of the
        # objective function.
        for i in range(num_dvars):
            linear_term = linear_terms[i]
            if linear_term != 0:
                self._add_term(linear_term, i)
        for i in range(num_dvars):
            for j in range(num_dvars):
                quadratic_term = quadratic_terms[i, j]
                if quadratic_term != 0:
                    self._add_term(quadratic_term, i, j)

        # This is technically optional and can wait until the optimizer is
        # constructed, but there's really no reason not to freeze immediately.
        self.freeze()

    def freeze(self):
        """Freeze the object to prevent further modification.

        Once an instance of this class is frozen, ``_add_variables`` and ``_add_term``
        can no longer be called.

        This operation is idempotent.  There is no way to undo it, as it exists
        to allow another object to rely on this one not changing its state
        going forward without having to make a copy as a distinct object.
        """
        if self._frozen is False:
            self._qubit_op = self._qubit_op.reduce()
        self._frozen = True

    @property
    def frozen(self) -> bool:
        """``True`` if the object can no longer be modified, ``False`` otherwise."""
        return self._frozen

    def ensure_thawed(self) -> None:
        """Raise a ``RuntimeError`` if the object is frozen and thus cannot be modified."""
        if self._frozen:
            raise RuntimeError("Cannot modify an encoding that has been frozen.")

    def get_state(self, dvars: Union[Dict[int, int], List[int]]):
        return get_problem_encoding_state(
            dvars, self.qubit_to_dvars, self.max_dvars_per_qubit
        )


class EncodingCommutationVerifier:
    """Class for verifying that the relaxation commutes with the objective function

    See also the "check encoding problem commutation" how-to notebook.
    """

    def __init__(self, encoding: QuantumRandomAccessEncoding):
        self._encoding = encoding

    def __len__(self) -> int:
        return 2**self._encoding.num_dvars

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, i: int) -> Tuple[str, float, float]:
        if i not in range(len(self)):
            raise IndexError(f"Index out of range: {i}")

        encoding = self._encoding
        str_dvars = ("{0:0" + str(encoding.num_dvars) + "b}").format(i)
        dvars = [int(b) for b in str_dvars]
        encoded_bitstr = encoding.get_state(dvars)

        # Offset accounts for the value of the encoded Hamiltonian's
        # identity coefficient. This term need not be evaluated directly as
        # Tr[I•rho] is always 1.
        offset = encoding.offset

        # Evaluate Un-encoded Problem
        # ========================
        # `sense` accounts for sign flips depending on whether
        # we are minimizing or maximizing the objective function
        problem = encoding.problem
        sense = problem.objective.sense.value
        obj_val = problem.objective.evaluate(dvars) * sense

        # Evaluate Encoded Problem
        # ========================
        encoded_problem = encoding.qubit_op  # H
        encoded_obj_val = (
            np.real((~StateFn(encoded_problem) @ encoded_bitstr).eval()) + offset
        )

        return (str_dvars, obj_val, encoded_obj_val)
