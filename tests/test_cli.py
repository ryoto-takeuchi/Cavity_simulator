from cavity_designer.cli import format_summary
from cavity_designer.io.parser import parse_cavity_config


def test_summary_shows_linear_round_trip_path_and_trace_metric():
    cavity = parse_cavity_config(
        {
            "wavelength": "1064 nm",
            "cavity": {"type": "linear"},
            "elements": [
                {"name": "M1", "type": "mirror"},
                {"name": "S12", "type": "space", "length": "26 mm"},
                {"name": "M2", "type": "curved_mirror", "Rc": "-24 mm"},
                {"name": "S23", "type": "space", "length": "48 mm"},
                {"name": "M3", "type": "curved_mirror", "Rc": "120 mm"},
                {"name": "S34", "type": "space", "length": "26 mm"},
                {"name": "M4", "type": "mirror"},
            ],
        }
    )

    summary = format_summary(cavity)

    assert "Trace metric |A+D|/2 =" in summary
    assert "Stable criterion: |A+D|/2 < 1" in summary
    assert "Reference plane: immediately after M1" in summary
    assert "Physical path: M1 -> S12 -> M2 -> S23 -> M3 -> S34 -> M4 -> S34 -> M3 -> S23 -> M2 -> S12 -> M1" in summary
    assert "ABCD sequence used: S12 -> M2 -> S23 -> M3 -> S34 -> M4 -> S34 -> M3 -> S23 -> M2 -> S12 -> M1" in summary


def test_summary_shows_ring_round_trip_path():
    cavity = parse_cavity_config(
        {
            "wavelength": "1064 nm",
            "cavity": {"type": "ring"},
            "elements": [
                {"name": "M1", "type": "mirror"},
                {"name": "S12", "type": "space", "length": "50 mm"},
                {"name": "M2", "type": "curved_mirror", "Rc": "200 mm"},
                {"name": "S21", "type": "space", "length": "50 mm"},
            ],
        }
    )

    summary = format_summary(cavity)

    assert "Physical path: M1 -> S12 -> M2 -> S21 -> M1" in summary
    assert "ABCD sequence used: S12 -> M2 -> S21 -> M1" in summary
