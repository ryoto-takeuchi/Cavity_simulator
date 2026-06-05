import pytest

from cavity_designer.core.cavity import Cavity
from cavity_designer.core.constants import SPEED_OF_LIGHT
from cavity_designer.core.elements import CurvedMirror, FlatMirror, Space
from cavity_designer.core.metrics import compute_metrics


def test_fsr_for_linear_cavity():
    cavity = Cavity(
        cavity_type="linear",
        wavelength=698e-9,
        elements=[FlatMirror("M1"), Space("space", 0.1), CurvedMirror("M2", radius_of_curvature=0.2)],
    )
    metrics = compute_metrics(cavity)
    assert metrics.fsr == pytest.approx(SPEED_OF_LIGHT / 0.2)


def test_fsr_for_ring_cavity():
    cavity = Cavity(
        cavity_type="ring",
        wavelength=698e-9,
        elements=[FlatMirror("M1"), Space("space", 0.1), CurvedMirror("M2", radius_of_curvature=0.2), Space("space2", 0.1)],
    )
    metrics = compute_metrics(cavity)
    assert metrics.fsr == pytest.approx(SPEED_OF_LIGHT / 0.2)


def test_finesse_and_linewidth_from_small_round_trip_loss():
    cavity = Cavity(
        cavity_type="linear",
        wavelength=698e-9,
        elements=[
            FlatMirror("M1", power_transmission_value=100e-6, power_loss_value=10e-6),
            Space("space", 0.1),
            CurvedMirror("M2", radius_of_curvature=0.2, power_transmission_value=10e-6, power_loss_value=10e-6),
        ],
    )
    metrics = compute_metrics(cavity)
    expected_loss = 130e-6
    assert metrics.round_trip_power_loss == pytest.approx(expected_loss)
    assert metrics.finesse == pytest.approx(2 * 3.141592653589793 / expected_loss)
    assert metrics.linewidth == pytest.approx(metrics.fsr / metrics.finesse)
    assert metrics.cavity_pole == pytest.approx(metrics.linewidth / 2)
    assert metrics.photon_lifetime == pytest.approx(1 / (2 * 3.141592653589793 * metrics.linewidth))
    assert metrics.power_buildup == pytest.approx(4 * 100e-6 / expected_loss**2)
    assert metrics.coupling_regime == "over-coupled"


def test_ring_buildup_uses_one_way_power_sum_not_coherent_field_sum():
    cavity = Cavity(
        cavity_type="ring",
        wavelength=698e-9,
        elements=[
            FlatMirror("M1", power_transmission_value=0.10, power_loss_value=0.0),
            Space("space1", 0.05),
            FlatMirror("M2", power_transmission_value=0.20, power_loss_value=0.0),
            Space("space2", 0.05),
        ],
    )

    metrics = compute_metrics(cavity)
    expected_survival = 0.90 * 0.80
    assert metrics.power_buildup == pytest.approx(0.10 / (1.0 - expected_survival))
    assert metrics.coupling_regime == "one-way traveling-wave power sum"
