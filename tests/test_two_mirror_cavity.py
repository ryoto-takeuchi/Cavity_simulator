from math import pi, sqrt

import pytest

from cavity_designer.core.cavity import Cavity
from cavity_designer.core.elements import CurvedMirror, FlatMirror, Space


def test_symmetric_two_mirror_cavity_waist_matches_analytic_formula():
    wavelength = 1064e-9
    length = 0.1
    radius = 0.2
    cavity = Cavity(
        cavity_type="linear",
        wavelength=wavelength,
        elements=[
            CurvedMirror(name="M1", radius_of_curvature=radius, power_transmission_value=1e-4, power_loss_value=1e-5),
            Space(name="space", length_m=length),
            CurvedMirror(name="M2", radius_of_curvature=radius, power_transmission_value=1e-5, power_loss_value=1e-5),
        ],
    )

    g1, g2 = cavity.two_mirror_g_parameters()
    assert 0 < g1 * g2 < 1

    mode = cavity.eigenmode("tangential")
    assert mode is not None
    expected_waist = sqrt((wavelength / pi) * sqrt(length * (2 * radius - length)) / 2)
    assert mode.waist_radius == pytest.approx(expected_waist, rel=1e-8)


def test_plano_concave_cavity_stable_in_both_planes():
    cavity = Cavity(
        cavity_type="linear",
        wavelength=698e-9,
        elements=[
            FlatMirror(name="M1", power_transmission_value=100e-6, power_loss_value=10e-6),
            Space(name="space", length_m=0.1),
            CurvedMirror(name="M2", radius_of_curvature=0.2, power_transmission_value=10e-6, power_loss_value=10e-6),
        ],
    )

    assert cavity.eigenmode("tangential") is not None
    assert cavity.eigenmode("sagittal") is not None
    assert cavity.stability("tangential").stable


def test_unstable_two_mirror_cavity_has_no_mode():
    cavity = Cavity(
        cavity_type="linear",
        wavelength=698e-9,
        elements=[
            FlatMirror(name="M1"),
            Space(name="space", length_m=0.3),
            CurvedMirror(name="M2", radius_of_curvature=0.2),
        ],
    )
    assert cavity.eigenmode("tangential") is None
