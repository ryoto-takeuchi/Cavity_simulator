import pytest

from cavity_designer.core.constants import SPEED_OF_LIGHT
from cavity_designer.core.metrics import compute_metrics
from cavity_designer.io.parser import load_cavity, parse_cavity_config


def test_three_mirror_ring_reports_separate_planes_for_oblique_curved_mirror():
    cavity = parse_cavity_config(
        {
            "wavelength": "698 nm",
            "cavity": {"type": "ring"},
            "elements": [
                {"name": "M1", "type": "mirror", "T": "150 ppm", "L": "10 ppm"},
                {"name": "S12", "type": "space", "length": "50 mm"},
                {"name": "M2", "type": "curved_mirror", "Rc": "200 mm", "AOI": "5 deg", "T": "20 ppm", "L": "10 ppm"},
                {"name": "S23", "type": "space", "length": "50 mm"},
                {"name": "M3", "type": "mirror", "T": "20 ppm", "L": "10 ppm"},
                {"name": "S31", "type": "space", "length": "50 mm"},
            ],
        }
    )
    tangential = cavity.eigenmode("tangential")
    sagittal = cavity.eigenmode("sagittal")

    assert tangential is not None
    assert sagittal is not None
    assert tangential.q != pytest.approx(sagittal.q)


def test_four_mirror_bow_tie_is_stable_and_astigmatic():
    cavity = parse_cavity_config(
        {
            "wavelength": "698 nm",
            "cavity": {"type": "ring"},
            "elements": [
                {"name": "M1", "type": "mirror", "T": "150 ppm", "L": "10 ppm"},
                {"name": "S12", "type": "space", "length": "40 mm"},
                {"name": "M2", "type": "curved_mirror", "Rc": "150 mm", "AOI": "5 deg", "T": "20 ppm", "L": "10 ppm"},
                {"name": "S23", "type": "space", "length": "10 mm"},
                {"name": "M3", "type": "curved_mirror", "Rc": "150 mm", "AOI": "5 deg", "T": "20 ppm", "L": "10 ppm"},
                {"name": "S34", "type": "space", "length": "40 mm"},
                {"name": "M4", "type": "mirror", "T": "20 ppm", "L": "10 ppm"},
                {"name": "S41", "type": "space", "length": "10 mm"},
            ],
        }
    )
    tangential = cavity.eigenmode("tangential")
    sagittal = cavity.eigenmode("sagittal")

    assert cavity.stability("tangential").stable
    assert cavity.stability("sagittal").stable
    assert tangential is not None
    assert sagittal is not None
    assert tangential.waist_radius != pytest.approx(sagittal.waist_radius)


def test_paper_ring_cavity_reproduces_reported_length_fsr_and_gouy_phase():
    cavity = load_cavity("examples/ring_cavity_paper.yaml")
    metrics = compute_metrics(cavity)

    assert metrics.geometric_round_trip_length == pytest.approx(0.198002)
    assert metrics.fsr == pytest.approx(SPEED_OF_LIGHT / 0.198002)
    assert metrics.fsr / 1e6 == pytest.approx(1514.088, rel=1e-6)
    assert cavity.round_trip_gouy_phase("sagittal") is not None
    assert cavity.round_trip_gouy_phase("tangential") is not None
    assert cavity.round_trip_gouy_phase("sagittal") * 180 / 3.141592653589793 == pytest.approx(107.98, abs=0.01)
    assert cavity.round_trip_gouy_phase("tangential") * 180 / 3.141592653589793 == pytest.approx(109.35, abs=0.01)
