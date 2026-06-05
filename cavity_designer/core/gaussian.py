"""Gaussian beam q-parameter utilities."""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, atan2, isfinite, pi, sqrt

import numpy as np

from cavity_designer.core.constants import ABS_TOL, REL_TOL
from cavity_designer.core.matrix import ABCDMatrix


class GaussianModeError(ValueError):
    """Raised when no physical Gaussian eigenmode exists."""


@dataclass(frozen=True)
class GaussianMode:
    q: complex
    wavelength: float

    @property
    def beam_radius(self) -> float:
        return beam_radius_from_q(self.q, self.wavelength)

    @property
    def wavefront_curvature(self) -> float:
        return wavefront_curvature_from_q(self.q)

    @property
    def rayleigh_range(self) -> float:
        return self.q.imag

    @property
    def waist_position(self) -> float:
        return -self.q.real

    @property
    def waist_radius(self) -> float:
        return sqrt(self.wavelength * self.rayleigh_range / pi)

    @property
    def gouy_phase(self) -> float:
        return gouy_phase_from_q(self.q)


def is_stable(matrix: ABCDMatrix, tol: float = ABS_TOL) -> bool:
    return abs(matrix.trace_half) < 1.0 - tol


def round_trip_gouy_phase(matrix: ABCDMatrix) -> float | None:
    """Return the stable round-trip Gouy phase advance in radians."""

    if not is_stable(matrix):
        return None
    trace_half = min(1.0, max(-1.0, matrix.trace_half))
    return acos(trace_half)


def propagate_q(q: complex, matrix: ABCDMatrix) -> complex:
    denominator = matrix.C * q + matrix.D
    if abs(denominator) < ABS_TOL:
        raise GaussianModeError("ABCD propagation encountered Cq + D = 0.")
    return (matrix.A * q + matrix.B) / denominator


def solve_eigenmode(matrix: ABCDMatrix, wavelength: float) -> GaussianMode | None:
    """Solve q = (A q + B)/(C q + D) and return the physical root."""

    if wavelength <= 0:
        raise GaussianModeError("Wavelength must be positive.")
    if not is_stable(matrix):
        return None

    roots: list[complex] = []
    if abs(matrix.C) < ABS_TOL:
        denominator = matrix.D - matrix.A
        if abs(denominator) < ABS_TOL:
            return None
        roots.append(matrix.B / denominator)
    else:
        coefficients = [matrix.C, matrix.D - matrix.A, -matrix.B]
        roots.extend(complex(root) for root in np.roots(coefficients))

    physical_roots = []
    for root in roots:
        if root.imag <= ABS_TOL:
            continue
        if not _finite_complex(root):
            continue
        try:
            radius = beam_radius_from_q(root, wavelength)
        except GaussianModeError:
            continue
        if not isfinite(radius) or radius <= 0:
            continue
        residual = propagate_q(root, matrix) - root
        scale = max(abs(root), 1.0)
        if abs(residual) > max(ABS_TOL, REL_TOL * scale * 100):
            continue
        physical_roots.append(root)

    if not physical_roots:
        return None
    physical_root = max(physical_roots, key=lambda value: value.imag)
    return GaussianMode(q=physical_root, wavelength=wavelength)


def beam_radius_from_q(q: complex, wavelength: float) -> float:
    inv_q = 1.0 / q
    imaginary = inv_q.imag
    if imaginary >= -ABS_TOL:
        raise GaussianModeError("q does not represent a physical Gaussian beam.")
    return sqrt(-wavelength / (pi * imaginary))


def wavefront_curvature_from_q(q: complex) -> float:
    real_part = (1.0 / q).real
    if abs(real_part) < ABS_TOL:
        return float("inf")
    return 1.0 / real_part


def gouy_phase_from_q(q: complex) -> float:
    return atan2(q.real, q.imag)


def _finite_complex(value: complex) -> bool:
    return isfinite(value.real) and isfinite(value.imag)
