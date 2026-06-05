from math import pi, sqrt

import pytest

from cavity_designer.core.gaussian import beam_radius_from_q, solve_eigenmode, wavefront_curvature_from_q
from cavity_designer.core.matrix import ABCDMatrix


def test_beam_radius_from_waist_q():
    wavelength = 1e-6
    z_rayleigh = 0.02
    q = 1j * z_rayleigh
    assert beam_radius_from_q(q, wavelength) == pytest.approx(sqrt(wavelength * z_rayleigh / pi))
    assert wavefront_curvature_from_q(q) == float("inf")


def test_stable_cavity_matrix_returns_physical_q():
    matrix = ABCDMatrix(-1, 0.1, -20, 1)
    mode = solve_eigenmode(matrix, 1e-6)
    assert mode is not None
    assert mode.q.imag > 0


def test_unstable_matrix_returns_none():
    matrix = ABCDMatrix(1, 0.1, 0, 1)
    assert solve_eigenmode(matrix, 1e-6) is None


def test_marginal_matrix_returns_none():
    matrix = ABCDMatrix(-1, 0, 0, -1)
    assert solve_eigenmode(matrix, 1e-6) is None
