import pytest

from cavity_designer.core.metrics import compute_metrics
from cavity_designer.core.elements import ThinLens
from cavity_designer.io.parser import load_cavity, parse_cavity_config


def test_linear_cavity_with_intracavity_lens_is_supported():
    cavity = load_cavity("examples/intracavity_lens.yaml")

    assert cavity.stability("tangential").stable
    assert cavity.stability("sagittal").stable
    assert cavity.eigenmode("tangential") is not None
    assert cavity.eigenmode("sagittal") is not None
    assert any(isinstance(element, ThinLens) and element.power_loss > 0 for element in cavity.elements)


def test_linear_intracavity_lens_loss_is_counted_twice_per_round_trip():
    cavity = parse_cavity_config(
        {
            "wavelength": "1064 nm",
            "cavity": {"type": "linear"},
            "elements": [
                {"name": "M1", "type": "mirror", "T": "100 ppm", "L": "10 ppm"},
                {"name": "S1", "type": "space", "length": "50 mm"},
                {"name": "L1", "type": "thin_lens", "f": "100 mm", "L": "200 ppm"},
                {"name": "S2", "type": "space", "length": "50 mm"},
                {"name": "M2", "type": "mirror", "T": "20 ppm", "L": "10 ppm"},
            ],
        }
    )
    metrics = compute_metrics(cavity)

    mirror_loss = 100e-6 + 10e-6 + 20e-6 + 10e-6
    lens_loss = 2 * 200e-6
    expected_loss = mirror_loss + lens_loss
    assert metrics.round_trip_power_loss == pytest.approx(expected_loss)
    assert metrics.power_buildup == pytest.approx(4 * 100e-6 / expected_loss**2)
    assert metrics.coupling_regime == "under-coupled"


def test_astigmatic_intracavity_lens_gives_plane_specific_modes():
    cavity = parse_cavity_config(
        {
            "wavelength": "1064 nm",
            "cavity": {"type": "linear"},
            "elements": [
                {"name": "M1", "type": "mirror", "T": "100 ppm", "L": "10 ppm"},
                {"name": "S1", "type": "space", "length": "50 mm"},
                {
                    "name": "L1",
                    "type": "thin_lens",
                    "f_tangential": "100 mm",
                    "f_sagittal": "150 mm",
                    "L": "50 ppm",
                },
                {"name": "S2", "type": "space", "length": "50 mm"},
                {"name": "M2", "type": "mirror", "T": "20 ppm", "L": "10 ppm"},
            ],
        }
    )
    tangential = cavity.eigenmode("tangential")
    sagittal = cavity.eigenmode("sagittal")

    assert tangential is not None
    assert sagittal is not None
    assert tangential.waist_radius != pytest.approx(sagittal.waist_radius)
    assert cavity.elements[2].power_loss == pytest.approx(50e-6)


def test_negative_focal_length_lens_is_parsed_for_defocusing_models():
    cavity = parse_cavity_config(
        {
            "wavelength": "1064 nm",
            "cavity": {"type": "linear"},
            "elements": [
                {"name": "M1", "type": "curved_mirror", "Rc": "200 mm"},
                {"name": "S1", "type": "space", "length": "20 mm"},
                {"name": "L1", "type": "thin_lens", "f": "-500 mm"},
                {"name": "S2", "type": "space", "length": "20 mm"},
                {"name": "M2", "type": "curved_mirror", "Rc": "200 mm"},
            ],
        }
    )

    lens = cavity.elements[2]
    assert lens.focal_length == pytest.approx(-0.5)


def test_ring_intracavity_lens_loss_is_counted_once_per_round_trip():
    cavity = parse_cavity_config(
        {
            "wavelength": "1064 nm",
            "cavity": {"type": "ring"},
            "elements": [
                {"name": "M1", "type": "mirror", "T": "100 ppm", "L": "10 ppm"},
                {"name": "S1", "type": "space", "length": "50 mm"},
                {"name": "L1", "type": "thin_lens", "f": "100 mm", "L": "75 ppm"},
                {"name": "S2", "type": "space", "length": "50 mm"},
                {"name": "M2", "type": "curved_mirror", "Rc": "200 mm", "T": "20 ppm", "L": "10 ppm"},
            ],
        }
    )
    metrics = compute_metrics(cavity)

    mirror_loss = 100e-6 + 10e-6 + 20e-6 + 10e-6
    assert metrics.round_trip_power_loss == pytest.approx(mirror_loss + 75e-6)


def test_negative_lens_loss_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        parse_cavity_config(
            {
                "wavelength": "1064 nm",
                "cavity": {"type": "linear"},
                "elements": [
                    {"name": "M1", "type": "mirror"},
                    {"name": "S1", "type": "space", "length": "50 mm"},
                    {"name": "L1", "type": "thin_lens", "f": "100 mm", "L": -0.1},
                    {"name": "S2", "type": "space", "length": "50 mm"},
                    {"name": "M2", "type": "mirror"},
                ],
            }
        )
