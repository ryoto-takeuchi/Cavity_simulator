import pytest
import numpy as np

from cavity_designer.core.layout import compute_cavity_layout
from cavity_designer.io.parser import parse_cavity_config
from cavity_designer.plot.layout_plot import _beam_envelope_points, _curved_mirror_points, plot_cavity_layout


def test_linear_retroreflection_layout_closes():
    cavity = parse_cavity_config(
        {
            "wavelength": "1064 nm",
            "layout": {"start": ["0 mm", "0 mm"], "direction": "0 deg"},
            "cavity": {"type": "linear"},
            "elements": [
                {"name": "M1", "type": "mirror", "AOI": "0 deg"},
                {"name": "S12", "type": "space", "length": "50 mm"},
                {"name": "M2", "type": "mirror", "AOI": "0 deg"},
            ],
        }
    )

    layout = compute_cavity_layout(cavity)

    assert layout.beam_points[0] == pytest.approx((0.0, 0.0))
    assert layout.beam_points[1] == pytest.approx((0.05, 0.0))
    assert layout.beam_points[-1] == pytest.approx((0.0, 0.0))
    assert layout.closure_error == pytest.approx(0.0, abs=1e-12)
    assert layout.direction_error == pytest.approx(0.0, abs=1e-12)


def test_signed_aoi_selects_opposite_layout_branches():
    def make_cavity(aoi: str):
        return parse_cavity_config(
            {
                "wavelength": "1064 nm",
                "layout": {"direction": "0 deg"},
                "cavity": {"type": "linear"},
                "elements": [
                    {"name": "M1", "type": "mirror", "AOI": "0 deg"},
                    {"name": "S12", "type": "space", "length": "50 mm"},
                    {"name": "M2", "type": "mirror", "AOI": aoi},
                    {"name": "S23", "type": "space", "length": "50 mm"},
                    {"name": "M3", "type": "mirror", "AOI": "0 deg"},
                ],
            }
        )

    positive = compute_cavity_layout(make_cavity("10 deg"))
    negative = compute_cavity_layout(make_cavity("-10 deg"))

    assert positive.beam_points[2][1] == pytest.approx(-negative.beam_points[2][1])
    assert positive.beam_points[2][1] != pytest.approx(0.0)


def test_negative_aoi_is_parsed_for_curved_mirror_layout():
    cavity = parse_cavity_config(
        {
            "wavelength": "1064 nm",
            "layout": {"direction": "0 deg"},
            "cavity": {"type": "linear"},
            "elements": [
                {"name": "M1", "type": "mirror"},
                {"name": "S12", "type": "space", "length": "50 mm"},
                {"name": "M2", "type": "curved_mirror", "Rc": "150 mm", "AOI": "-5 deg"},
            ],
        }
    )

    assert cavity.elements[-1].angle_of_incidence == pytest.approx(-5 * 3.141592653589793 / 180)


def test_layout_plot_is_created(tmp_path):
    cavity = parse_cavity_config(
        {
            "wavelength": "1064 nm",
            "layout": {"direction": "0 deg"},
            "cavity": {"type": "linear"},
            "elements": [
                {"name": "M1", "type": "mirror"},
                {"name": "S12", "type": "space", "length": "50 mm"},
                {"name": "M2", "type": "curved_mirror", "Rc": "150 mm", "AOI": "0 deg"},
            ],
        }
    )
    output = plot_cavity_layout(cavity, tmp_path / "layout.png")

    assert output.exists()
    assert output.stat().st_size > 0


def test_layout_mirror_size_is_parsed():
    cavity = parse_cavity_config(
        {
            "wavelength": "1064 nm",
            "layout": {"mirror_size": "25 mm", "beam_radius_scale": 40},
            "cavity": {"type": "linear"},
            "elements": [
                {"name": "M1", "type": "mirror"},
                {"name": "S12", "type": "space", "length": "50 mm"},
                {"name": "M2", "type": "mirror"},
            ],
        }
    )

    assert cavity.layout.mirror_size == pytest.approx(0.025)
    assert cavity.layout.beam_radius_scale == pytest.approx(40)


def test_curved_mirror_plot_points_use_signed_roc_sag():
    position = np.array([0.0, 0.0])
    positive = _curved_mirror_points(position, 0.0, 0.05, 0.02)
    negative = _curved_mirror_points(position, 0.0, -0.05, 0.02)

    assert positive[0, 0] == pytest.approx(-0.01)
    assert positive[-1, 0] == pytest.approx(0.01)
    assert positive[len(positive) // 2, 1] == pytest.approx(0.0)
    assert positive[0, 1] > 0.0
    assert negative[0, 1] == pytest.approx(-positive[0, 1])


def test_large_roc_draws_flatter_arc_than_small_roc():
    position = np.array([0.0, 0.0])
    small = _curved_mirror_points(position, 0.0, 0.05, 0.02)
    large = _curved_mirror_points(position, 0.0, 0.5, 0.02)

    assert large[0, 1] < small[0, 1]


def test_beam_envelope_width_uses_gaussian_radius_and_layout_scale():
    cavity = parse_cavity_config(
        {
            "wavelength": "1064 nm",
            "layout": {"beam_radius_scale": 25},
            "cavity": {"type": "linear"},
            "elements": [
                {"name": "M1", "type": "mirror"},
                {"name": "S12", "type": "space", "length": "100 mm"},
                {"name": "M2", "type": "curved_mirror", "Rc": "200 mm"},
            ],
        }
    )
    layout = compute_cavity_layout(cavity)
    envelope = _beam_envelope_points(cavity, layout.beam_points, "tangential", samples_per_space=20)

    assert envelope is not None
    upper, lower = envelope
    expected_width = 2 * cavity.eigenmode("tangential").beam_radius * cavity.layout.beam_radius_scale
    assert np.linalg.norm(upper[0] - lower[0]) == pytest.approx(expected_width)
