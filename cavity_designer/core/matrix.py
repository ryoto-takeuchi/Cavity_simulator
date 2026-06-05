"""ABCD ray-transfer matrix utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from cavity_designer.core.constants import ABS_TOL


@dataclass(frozen=True)
class ABCDMatrix:
    """A 2x2 ray-transfer matrix.

    Multiplication order follows optical propagation order: if light sees
    matrices M1, then M2, the combined matrix is M2 @ M1.
    """

    A: float
    B: float
    C: float
    D: float

    @classmethod
    def identity(cls) -> "ABCDMatrix":
        return cls(1.0, 0.0, 0.0, 1.0)

    @classmethod
    def from_array(cls, array: np.ndarray) -> "ABCDMatrix":
        if array.shape != (2, 2):
            raise ValueError("ABCD matrix array must have shape (2, 2).")
        return cls(float(array[0, 0]), float(array[0, 1]), float(array[1, 0]), float(array[1, 1]))

    def as_array(self) -> np.ndarray:
        return np.array([[self.A, self.B], [self.C, self.D]], dtype=float)

    @property
    def trace_half(self) -> float:
        return 0.5 * (self.A + self.D)

    @property
    def determinant(self) -> float:
        return self.A * self.D - self.B * self.C

    def then(self, next_matrix: "ABCDMatrix") -> "ABCDMatrix":
        """Return the matrix for this transform followed by next_matrix."""

        return ABCDMatrix.from_array(next_matrix.as_array() @ self.as_array())


def compose(matrices: Iterable[ABCDMatrix]) -> ABCDMatrix:
    """Compose matrices in the order encountered by the optical beam."""

    total = np.eye(2)
    for matrix in matrices:
        total = matrix.as_array() @ total
    return ABCDMatrix.from_array(total)


def space_matrix(length: float) -> ABCDMatrix:
    if length < 0:
        raise ValueError("Space length must be non-negative.")
    return ABCDMatrix(1.0, float(length), 0.0, 1.0)


def thin_lens_matrix(focal_length: float) -> ABCDMatrix:
    if abs(focal_length) < ABS_TOL:
        raise ValueError("Thin lens focal length must be non-zero.")
    return ABCDMatrix(1.0, 0.0, -1.0 / focal_length, 1.0)


def flat_mirror_matrix() -> ABCDMatrix:
    return ABCDMatrix.identity()


def curved_mirror_reflection_matrix(radius_of_curvature: float) -> ABCDMatrix:
    """Reflection matrix for a spherical curved mirror at normal incidence."""

    if np.isinf(radius_of_curvature):
        return flat_mirror_matrix()
    if abs(radius_of_curvature) < ABS_TOL:
        raise ValueError("Mirror radius of curvature must be non-zero.")
    return ABCDMatrix(1.0, 0.0, -2.0 / radius_of_curvature, 1.0)


def custom_matrix(A: float, B: float, C: float, D: float) -> ABCDMatrix:
    return ABCDMatrix(float(A), float(B), float(C), float(D))
