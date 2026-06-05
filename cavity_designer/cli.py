"""Command-line interface for cavity_designer."""

from __future__ import annotations

import argparse
from math import degrees
import sys
from pathlib import Path

from cavity_designer.core.cavity import CavityError
from cavity_designer.core.elements import PLANES
from cavity_designer.core.metrics import compute_metrics
from cavity_designer.io.parser import ConfigError, load_cavity
from cavity_designer.plot.beam_plot import plot_beam_radius, plot_wavefront_curvature
from cavity_designer.plot.layout_plot import plot_cavity_layout
from cavity_designer.plot.stability_plot import plot_g1_g2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Design and inspect Gaussian/ABCD optical cavities.")
    parser.add_argument("config", help="YAML cavity configuration file")
    parser.add_argument("--summary", action="store_true", help="Print a text summary")
    parser.add_argument("--plot", action="store_true", help="Save beam and stability plots")
    parser.add_argument("--output", default="outputs/cavity", help="Output directory for plots")
    parser.add_argument("--samples-per-space", type=int, default=120, help="Sampling density for each space")
    args = parser.parse_args(argv)

    try:
        cavity = load_cavity(args.config)
        should_print_summary = args.summary or not args.plot
        if should_print_summary:
            print(format_summary(cavity))
        if args.plot:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            plot_cavity_layout(cavity, output_dir / "layout.png", samples_per_space=args.samples_per_space)
            plot_beam_radius(cavity, output_dir / "beam_radius.png", samples_per_space=args.samples_per_space)
            plot_wavefront_curvature(cavity, output_dir / "wavefront_curvature.png", samples_per_space=args.samples_per_space)
            g_values = cavity.two_mirror_g_parameters()
            if g_values is not None:
                plot_g1_g2(output_dir / "g1_g2.png", *g_values)
            print(f"Saved plots to {output_dir}")
    except (ConfigError, CavityError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


def format_summary(cavity) -> str:
    metrics = compute_metrics(cavity)
    lines: list[str] = []
    lines.extend(
        [
            "Cavity summary",
            "--------------",
            f"Type: {cavity.cavity_type}",
            f"Wavelength: {cavity.wavelength * 1e9:.3f} nm",
            f"Round-trip length: {metrics.geometric_round_trip_length:.6f} m ({metrics.geometric_round_trip_length * 1e3:.3f} mm)",
            f"FSR: {_format_frequency(metrics.fsr)}",
            "",
            "Round-trip path",
            "---------------",
            f"Reference plane: immediately after {cavity.elements[0].name}",
            f"Physical path: {_format_physical_path(cavity)}",
            f"ABCD sequence used: {_format_abcd_sequence(cavity)}",
            "",
        ]
    )

    for plane in PLANES:
        stability = cavity.stability(plane)
        matrix = stability.matrix
        lines.extend(
            [
                f"Stability ({plane})",
                "-" * (12 + len(plane)),
                "Round-trip ABCD:",
                f"A = {matrix.A:.9g}",
                f"B = {matrix.B:.9g}",
                f"C = {matrix.C:.9g}",
                f"D = {matrix.D:.9g}",
                f"Trace metric |A+D|/2 = {abs(matrix.A + matrix.D) / 2.0:.9g}",
                "Stable criterion: |A+D|/2 < 1",
                f"Stable: {'yes' if stability.stable else 'no'}",
            ]
        )
        mode = cavity.eigenmode(plane)
        lines.append("")
        lines.append(f"Gaussian mode ({plane})")
        lines.append("-" * (16 + len(plane)))
        if mode is None:
            lines.append("No stable Gaussian eigenmode exists for this plane.")
        else:
            lines.extend(
                [
                    f"q at reference plane: {mode.q.real:.9g} + {mode.q.imag:.9g}j m",
                    f"beam radius at reference: {mode.beam_radius * 1e6:.3f} um",
                    f"waist radius: {mode.waist_radius * 1e6:.3f} um",
                    f"waist position: {mode.waist_position * 1e3:.3f} mm from reference",
                    f"Rayleigh range: {mode.rayleigh_range * 1e3:.3f} mm",
                ]
            )
        lines.append("")

    g_values = cavity.two_mirror_g_parameters()
    if g_values is not None:
        g1, g2 = g_values
        lines.extend(
            [
                "Two-mirror diagnostics",
                "----------------------",
                f"g1 = {g1:.9g}",
                f"g2 = {g2:.9g}",
                f"0 < g1*g2 < 1: {'yes' if 0.0 < g1 * g2 < 1.0 else 'no'}",
                "",
            ]
        )

    lines.extend(
        [
            "RoundTrip Gouy phase shift",
            "--------------------------",
        ]
    )
    for plane in ("sagittal", "tangential"):
        phase = cavity.round_trip_gouy_phase(plane)
        label = "Sagittal" if plane == "sagittal" else "Tangential"
        if phase is None:
            lines.append(f"{label}: not stable")
        else:
            lines.append(f"{label}: {degrees(phase):.2f} degree")
    lines.append("")

    lines.extend(
        [
            "Cavity metrics",
            "--------------",
            f"Round-trip power loss: {metrics.round_trip_power_loss * 1e6:.3f} ppm",
            f"Finesse: {_format_optional(metrics.finesse)}",
            f"Linewidth: {_format_optional_frequency(metrics.linewidth)}",
            f"Cavity pole: {_format_optional_frequency(metrics.cavity_pole)}",
            f"Photon lifetime: {_format_optional_time(metrics.photon_lifetime)}",
            f"Estimated build-up: {_format_optional(metrics.power_buildup)}",
            f"Coupling regime: {metrics.coupling_regime}",
        ]
    )
    for warning in metrics.warnings:
        lines.append(f"Warning: {warning}")
    return "\n".join(lines)


def _format_physical_path(cavity) -> str:
    names = [element.name for element in cavity.elements[:1] + cavity.sequence_for_round_trip()]
    return " -> ".join(names)


def _format_abcd_sequence(cavity) -> str:
    return " -> ".join(element.name for element in cavity.sequence_for_round_trip())


def _format_optional(value: float | None) -> str:
    if value is None:
        return "not estimated"
    return f"{value:.6g}"


def _format_frequency(value: float) -> str:
    if abs(value) >= 10e9:
        return f"{value / 1e9:.3f} GHz"
    if abs(value) >= 1e6:
        return f"{value / 1e6:.3f} MHz"
    if abs(value) >= 1e3:
        return f"{value / 1e3:.3f} kHz"
    return f"{value:.3f} Hz"


def _format_optional_frequency(value: float | None) -> str:
    if value is None:
        return "not estimated"
    return _format_frequency(value)


def _format_optional_time(value: float | None) -> str:
    if value is None:
        return "not estimated"
    if abs(value) >= 1e-3:
        return f"{value * 1e3:.3f} ms"
    if abs(value) >= 1e-6:
        return f"{value * 1e6:.3f} us"
    if abs(value) >= 1e-9:
        return f"{value * 1e9:.3f} ns"
    return f"{value:.3e} s"


if __name__ == "__main__":
    raise SystemExit(main())
