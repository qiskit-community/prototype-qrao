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

   z_to_31p_qrac_basis_circuit
   z_to_21p_qrac_basis_circuit
   qrac_state_prep_1q
   qrac_state_prep_multiqubit
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


def to_n1p_qrac_basis(n_dvars, basis) -> QuantumCircuit:
    """Return the basis change corresponding to the (n_dvars, 1, p)-QRAC
       for the given basis

    Args:
        n: The number of decision variables encoded in the qubit.
        state: The index of the (n_dvars, 1, p)-QRAC basis to change to.

    Returns:
        The ``QuantumCircuit`` implementing the change of basis.
    """

    if n_dvars not in (1, 2, 3):
        raise ValueError(f"n_dvars must be 1, 2, or 3, not {n_dvars}.")
    n_states = 2 ** (n_dvars - 1)
    if basis not in range(0, n_states):
        raise ValueError(f"state must be in [0, {n_states}], not {basis}.")

    beta = np.arccos(1 / np.sqrt(n_dvars))

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


def state_from_dvar_values(*dvar_values: int) -> CircuitStateFn:
    """Prepare a single-qubit QRAC state from a list of decision variables.

      This function accepts 1, 2, or 3 decision variables, in which case it
      generates a 1-QRAC, 2-QRAC, or 3-QRAC, respectively.

    Args:
        dvars: The values of the decision variables to encode. Each decision 
               variable must have value 0 or 1.

    Returns:
        The single-qubit QRAC circuit state function.

    """
    n_dvars = len(dvar_values)

    if n_dvars not in (1, 2, 3):
        raise TypeError(
            f"state_from_dvars can take up to 3 decision variables, not {n_dvars}."
        )
    if not all(dvar_value in (0, 1) for dvar_value in dvar_values):
        raise ValueError("Each decision variable must have value 0 or 1.")

    even_parity = sum(dvar_values) % 2
    even_count = n_dvars % 2

    basis = sum(
        [(2**i) * (dvar_values[i] ^ dvar_values[n_dvars - 1]) for i in range(n_dvars - 1)]
    )
    state = One if (even_parity if even_count else dvar_values[0]) else Zero
    return (CircuitOp(to_n1p_qrac_basis(n_dvars, basis)) @ state).to_circuit_op()


def qrac_state_prep_multiqubit(
    dvars: Union[Dict[int, int], List[int]],
    q2vars: List[List[int]],
    max_vars_per_qubit: int,
) -> CircuitStateFn:
    remaining_dvars = set(dvars if isinstance(dvars, dict) else range(len(dvars)))
    ordered_bits = []
    for qi_vars in q2vars:
        if len(qi_vars) > max_vars_per_qubit:
            raise ValueError(
                "Each qubit is expected to be associated with at most "
                f"`max_vars_per_qubit` ({max_vars_per_qubit}) variables, "
                f"not {len(qi_vars)} variables."
            )
        if not qi_vars:
            # This probably actually doesn't cause any issues, but why support
            # it (and test this edge case) if we don't have to?
            raise ValueError(
                "There is a qubit without any decision variables assigned to it."
            )
        qi_bits: List[int] = []
        for dv in qi_vars:
            try:
                qi_bits.append(dvars[dv])
            except (KeyError, IndexError):
                raise ValueError(
                    f"Decision variable not included in dvars: {dv}"
                ) from None
            try:
                remaining_dvars.remove(dv)
            except KeyError:
                raise ValueError(
                    f"Unused decision variable(s) in dvars: {remaining_dvars}"
                ) from None
        ordered_bits.append(qi_bits)

    if remaining_dvars:
        raise ValueError(f"Not all dvars were included in q2vars: {remaining_dvars}")

    qracs = [state_from_dvar_values(*qi_bits) for qi_bits in ordered_bits]
    logical = reduce(lambda x, y: x ^ y, qracs)
    return logical


def q2vars_from_var2op(var2op: Dict[int, Tuple[int, PrimitiveOp]]) -> List[List[int]]:
    """Calculate q2vars given var2op"""
    num_qubits = max(qubit_index for qubit_index, _ in var2op.values()) + 1
    q2vars: List[List[int]] = [[] for i in range(num_qubits)]
    for var, (q, _) in var2op.items():
        q2vars[q].append(var)
    return q2vars


class QuantumRandomAccessEncoding:
    """This class specifies a Quantum Random Access Code that can be used to encode
    the binary variables of a QUBO (quadratic unconstrained binary optimization
    problem).

    Args:
        max_vars_per_qubit: maximum possible compression ratio.
            Supported values are 1, 2, or 3.

    """

    # This defines the convention of the Pauli operators (and their ordering)
    # for each encoding.
    OPERATORS = (
        (Z,),  # (1,1,1) QRAC
        (X, Z),  # (2,1,p) QRAC, p ≈ 0.85
        (X, Y, Z),  # (3,1,p) QRAC, p ≈ 0.79
    )

    def __init__(self, max_vars_per_qubit: int = 3):
        if max_vars_per_qubit not in (1, 2, 3):
            raise ValueError("max_vars_per_qubit must be 1, 2, or 3")
        self._max_vars_per_qubit = max_vars_per_qubit

        self._qubit_op: Optional[Union[PauliOp, PauliSumOp]] = None
        self._offset: Optional[float] = None
        self._problem: Optional[QuadraticProgram] = None
        self._var2op: Dict[int, Tuple[int, PrimitiveOp]] = {}
        self._q2vars: List[List[int]] = []
        self._frozen = False

    @property
    def num_qubits(self) -> int:
        return len(self._q2vars)

    @property
    def num_vars(self) -> int:
        return len(self._var2op)

    @property
    def max_vars_per_qubit(self) -> int:
        return self._max_vars_per_qubit

    @property
    def var2op(self) -> Dict[int, Tuple[int, PrimitiveOp]]:
        return self._var2op

    @property
    def q2vars(self) -> List[List[int]]:
        return self._q2vars

    @property
    def compression_ratio(self) -> float:
        return self.num_vars / self.num_qubits

    @property
    def minimum_recovery_probability(self) -> float:
        n = self.max_vars_per_qubit
        return (1 + 1 / np.sqrt(n)) / 2

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

    def _add_variables(self, variables: List[int]) -> None:
        self.ensure_thawed()
        # NOTE: If this is called multiple times, it *always* adds an
        # additional qubit (see final line), even if aggregating them into a
        # single call would have resulted in fewer qubits.
        if self._qubit_op is not None:
            raise RuntimeError(
                "_add_variables() cannot be called once terms have been added "
                "to the operator, as the number of qubits must thereafter "
                "remain fixed."
            )
        if not variables:
            return
        if len(variables) != len(set(variables)):
            raise ValueError("Added variables must be unique")
        for v in variables:
            if v in self._var2op:
                raise ValueError("Added variables cannot collide with existing ones")
        # Modify the object now that error checking is complete.
        n = self.max_vars_per_qubit
        old_num_qubits = len(self._q2vars)
        num_new_qubits = _ceildiv(len(variables), n)
        # Populate self._var2op and self._q2vars
        for _ in range(num_new_qubits):
            self._q2vars.append([])
        for i, v in enumerate(variables):
            qubit, op = divmod(i, n)
            qubit_index = old_num_qubits + qubit
            self._q2vars[qubit_index].append(v)
        for i, v in enumerate(variables):
            qubit, op = divmod(i, n)
            qubit_index = old_num_qubits + qubit
            assert v not in self._var2op  # was checked above
            self._var2op[v] = (
                qubit_index,
                self.OPERATORS[len(self._q2vars[qubit_index]) - 1][op],
            )

    def _add_term(self, w: float, *variables: int) -> None:
        self.ensure_thawed()
        # Eq. (31) in https://arxiv.org/abs/2111.03167v2 assumes a weight-2
        # Pauli operator.  To generalize, we replace the `d` in that equation
        # with `d_prime`, defined as follows:
        d_prime = np.sqrt(
            np.prod([len(self.q2vars[self.var2op[x][0]]) for x in variables])
        )
        op = w * d_prime * self.term2op(*variables)
        # We perform the following short-circuit *after* calling term2op so at
        # least we have confirmed that the user provided a valid variables list.
        if w == 0.0:
            return
        if self._qubit_op is None:
            self._qubit_op = op
        else:
            self._qubit_op += op

    def term2op(self, *variables: int) -> PauliOp:
        ops = [I] * self.num_qubits
        done = set()
        for x in variables:
            pos, op = self._var2op[x]
            if pos in done:
                raise RuntimeError(f"Collision of variables: {variables}")
            ops[pos] = op
            done.add(pos)
        return reduce(lambda x, y: x ^ y, ops)

    @staticmethod
    def _find_variable_partition(quad: np.ndarray) -> Dict[int, List[int]]:
        num_nodes = quad.shape[0]
        assert quad.shape == (num_nodes, num_nodes)
        graph = rx.PyGraph()
        graph.add_nodes_from(range(num_nodes))
        graph.add_edges_from_no_data(list(zip(*np.where(quad != 0))))
        node2color = rx.graph_greedy_color(graph)
        color2node: Dict[int, List[int]] = {}
        for node, color in sorted(node2color.items()):
            color2node.setdefault(color, []).append(node)
        return color2node

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
        # Ensure fresh object
        if self.num_qubits > 0:
            raise RuntimeError(
                "Must call encode() on an Encoding that has not been used already"
            )

        # if problem has variables that are not binary, raise an error
        if problem.get_num_vars() > problem.get_num_binary_vars():
            raise RuntimeError(
                "The type of all variables must be binary. "
                "You can use `QuadraticProgramToQubo` converter "
                "to convert integer variables to binary variables. "
                "If the problem contains continuous variables, `qrao` "
                "cannot handle it."
            )

        # if constraints exist, raise an error
        if problem.linear_constraints or problem.quadratic_constraints:
            raise RuntimeError(
                "There must be no constraint in the problem. "
                "You can use `QuadraticProgramToQubo` converter to convert "
                "constraints to penalty terms of the objective function."
            )

        # initialize Hamiltonian.
        num_vars = problem.get_num_vars()

        # set a sign corresponding to a maximized or minimized problem:
        # 1 is for minimized problem, -1 is for maximized problem.
        sense = problem.objective.sense.value

        # convert a constant part of the objective function into Hamiltonian.
        offset = problem.objective.constant * sense

        # convert linear parts of the objective function into Hamiltonian.
        linear = np.zeros(num_vars)
        for idx, coef in problem.objective.linear.to_dict().items():
            weight = coef * sense / 2
            linear[idx] -= weight
            offset += weight

        # convert quadratic parts of the objective function into Hamiltonian.
        quad = np.zeros((num_vars, num_vars))
        for (i, j), coef in problem.objective.quadratic.to_dict().items():
            weight = coef * sense / 4
            if i == j:
                linear[i] -= 2 * weight
                offset += 2 * weight
            else:
                quad[i, j] += weight
                linear[i] -= weight
                linear[j] -= weight
                offset += weight

        # Find variable partition (a graph coloring is sufficient)
        variable_partition = self._find_variable_partition(quad)

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
        assert sorted(chain.from_iterable(variable_partition.values())) == list(
            range(num_vars)
        )

        # generate a Hamiltonian
        for _, v in sorted(variable_partition.items()):
            self._add_variables(sorted(v))
        for i in range(num_vars):
            w = linear[i]
            if w != 0:
                self._add_term(w, i)
        for i in range(num_vars):
            for j in range(num_vars):
                w = quad[i, j]
                if w != 0:
                    self._add_term(w, i, j)

        self._offset = offset
        self._problem = problem

        # This is technically optional and can wait until the optimizer is
        # constructed, but there's really no reason not to freeze
        # immediately.
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
            raise RuntimeError("Cannot modify an encoding that has been frozen")

    def state_prep(self, dvars: Union[Dict[int, int], List[int]]):
        return qrac_state_prep_multiqubit(dvars, self.q2vars, self.max_vars_per_qubit)


class EncodingCommutationVerifier:
    """Class for verifying that the relaxation commutes with the objective function

    See also the "check encoding problem commutation" how-to notebook.
    """

    def __init__(self, encoding: QuantumRandomAccessEncoding):
        self._encoding = encoding

    def __len__(self) -> int:
        return 2**self._encoding.num_vars

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, i: int) -> Tuple[str, float, float]:
        if i not in range(len(self)):
            raise IndexError(f"Index out of range: {i}")

        encoding = self._encoding
        str_dvars = ("{0:0" + str(encoding.num_vars) + "b}").format(i)
        dvars = [int(b) for b in str_dvars]
        encoded_bitstr = encoding.state_prep(dvars)

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
