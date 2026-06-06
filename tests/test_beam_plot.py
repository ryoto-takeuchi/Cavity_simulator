from cavity_designer.io.parser import parse_cavity_config
from cavity_designer.plot.beam_plot import plot_beam_radius, plot_wavefront_curvature


def test_beam_plots_skip_unstable_plane_and_save_stable_plane(tmp_path):
    cavity = parse_cavity_config(
        {
            "wavelength": "698 nm",
            "cavity": {"type": "ring"},
            "elements": [
                {"name": "M1", "type": "mirror", "AOI": "-5 deg"},
                {"name": "S12", "type": "space", "length": "150 mm"},
                {"name": "M2", "type": "curved_mirror", "Rc": "150 mm", "AOI": "-5 deg"},
                {"name": "S23", "type": "space", "length": "150 mm"},
                {"name": "M3", "type": "curved_mirror", "Rc": "150 mm", "AOI": "5 deg"},
                {"name": "S34", "type": "space", "length": "150 mm"},
                {"name": "M4", "type": "mirror", "AOI": "5 deg"},
                {"name": "S41", "type": "space", "length": "150 mm"},
            ],
        }
    )

    beam_path = plot_beam_radius(cavity, tmp_path / "beam_radius.png")
    curvature_path = plot_wavefront_curvature(cavity, tmp_path / "wavefront_curvature.png")

    assert beam_path.exists()
    assert beam_path.stat().st_size > 0
    assert curvature_path.exists()
    assert curvature_path.stat().st_size > 0
