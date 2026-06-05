import pytest

from cavity_designer.io.parser import ConfigError, parse_angle, parse_cavity_config, parse_length, parse_power


def test_power_units():
    assert parse_power("100 ppm", "T") == pytest.approx(100e-6)
    assert parse_power("1 %", "T") == pytest.approx(0.01)
    assert parse_power(0.25, "T") == pytest.approx(0.25)


def test_length_units():
    assert parse_length("698 nm", "wavelength") == pytest.approx(698e-9)
    assert parse_length("100 mm", "length") == pytest.approx(0.1)


def test_angle_units_require_explicit_unit():
    assert parse_angle("5 deg", "AOI") == pytest.approx(0.08726646259971647)
    with pytest.raises(ConfigError, match="requires an explicit unit"):
        parse_angle(5, "AOI")


def test_invalid_unit():
    with pytest.raises(ConfigError, match="Invalid length unit"):
        parse_length("100 inch", "length")


def test_bare_length_number_rejected():
    with pytest.raises(ConfigError, match="requires an explicit unit"):
        parse_length(100, "length")


def test_missing_required_field():
    with pytest.raises(ConfigError, match="wavelength"):
        parse_cavity_config({"cavity": {"type": "linear"}, "elements": []})


def test_negative_loss_rejected():
    config = {
        "wavelength": "698 nm",
        "elements": [{"name": "M1", "type": "mirror", "L": -0.1}],
    }
    with pytest.raises(ConfigError, match="non-negative"):
        parse_cavity_config(config)


def test_transmission_plus_loss_greater_than_one_rejected():
    config = {
        "wavelength": "698 nm",
        "elements": [{"name": "M1", "type": "mirror", "T": 0.8, "L": 0.3}],
    }
    with pytest.raises(ValueError, match="negative reflectivity"):
        parse_cavity_config(config)


def test_negative_curved_mirror_radius_is_allowed_for_convex_mirror():
    cavity = parse_cavity_config(
        {
            "wavelength": "1064 nm",
            "cavity": {"type": "linear"},
            "elements": [
                {"name": "M1", "type": "mirror"},
                {"name": "space", "type": "space", "length": "50 mm"},
                {"name": "M2", "type": "curved_mirror", "Rc": "-150 mm"},
            ],
        }
    )
    mirror = cavity.elements[-1]
    assert mirror.radius_of_curvature == pytest.approx(-0.15)
    assert mirror.matrix().C == pytest.approx(2 / 0.15)


def test_mirror_aoi_defaults_to_zero_when_omitted():
    cavity = parse_cavity_config(
        {
            "wavelength": "1064 nm",
            "cavity": {"type": "linear"},
            "elements": [
                {"name": "M1", "type": "mirror"},
                {"name": "space", "type": "space", "length": "50 mm"},
                {"name": "M2", "type": "curved_mirror", "Rc": "150 mm"},
            ],
        }
    )
    assert cavity.elements[0].angle_of_incidence == pytest.approx(0.0)
    assert cavity.elements[-1].angle_of_incidence == pytest.approx(0.0)
