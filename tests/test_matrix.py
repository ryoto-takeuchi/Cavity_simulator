import numpy as np
import pytest

from cavity_designer.core.elements import CurvedMirror
from cavity_designer.core.matrix import compose, curved_mirror_reflection_matrix, space_matrix, thin_lens_matrix


def test_space_matrix():
    assert np.allclose(space_matrix(0.1).as_array(), [[1, 0.1], [0, 1]])


def test_thin_lens_matrix():
    assert np.allclose(thin_lens_matrix(0.2).as_array(), [[1, 0], [-5, 1]])


def test_curved_mirror_reflection_matrix():
    assert np.allclose(curved_mirror_reflection_matrix(0.25).as_array(), [[1, 0], [-8, 1]])


def test_convex_curved_mirror_reflection_matrix():
    assert np.allclose(curved_mirror_reflection_matrix(-0.25).as_array(), [[1, 0], [8, 1]])


def test_matrix_multiplication_order():
    matrix = compose([space_matrix(0.1), thin_lens_matrix(0.2)])
    expected = thin_lens_matrix(0.2).as_array() @ space_matrix(0.1).as_array()
    assert np.allclose(matrix.as_array(), expected)


def test_oblique_curved_mirror_splits_tangential_and_sagittal_power():
    mirror = CurvedMirror(name="M", radius_of_curvature=0.2, angle_of_incidence=np.deg2rad(60))

    tangential = mirror.matrix("tangential")
    sagittal = mirror.matrix("sagittal")

    assert tangential.C == pytest.approx(-2 / (0.2 * np.cos(np.deg2rad(60))))
    assert sagittal.C == pytest.approx(-2 * np.cos(np.deg2rad(60)) / 0.2)
