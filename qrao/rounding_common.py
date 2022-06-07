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

"""Common classes for rounding schemes"""

from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from qiskit.opflow import PrimitiveOp

from .encoding import get_qubit_from_op_assignment


@dataclass
class RoundingSolutionSample:
    """Partial SolutionSample for use in rounding results"""

    x: np.ndarray
    probability: float


class RoundingContext:
    """Information that is provided for rounding"""

    def __init__(
        self,
        dvar_to_op: Dict[int, Tuple[int, PrimitiveOp]],
        *,
        qubit_to_dvars: Optional[List[List[int]]] = None,
        trace_values=None,
        circuit=None
    ):
        self.dvar_to_op = dvar_to_op
        self.qubit_to_dvars = (
            get_qubit_from_op_assignment(dvar_to_op)
            if qubit_to_dvars is None
            else qubit_to_dvars
        )

        self.trace_values = trace_values  # TODO: rename me
        self.circuit = circuit  # TODO: rename me


class RoundingResult:
    """Base class for a rounding result"""

    def __init__(self, samples: List[RoundingSolutionSample], *, time_taken=None):
        self._samples = samples
        self.time_taken = time_taken

    @property
    def samples(self) -> List[RoundingSolutionSample]:
        return self._samples


class RoundingScheme(ABC):
    """Base class for a rounding scheme"""

    @abstractmethod
    def round(self, ctx: RoundingContext) -> RoundingResult:
        """Perform rounding

        Returns: an instance of RoundingResult
        """
