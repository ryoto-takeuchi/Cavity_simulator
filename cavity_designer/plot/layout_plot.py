"""2D cavity layout plot."""

from __future__ import annotations

from math import cos, isfinite, sin
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cavity_designer.core.cavity import Cavity
from cavity_designer.core.elements import Plane
from cavity_designer.core.layout import CavityLayout, LayoutOptic, compute_cavity_layout


def plot_cavity_layout(
    cavity: Cavity,
    output_path: str | Path,
    plane: Plane = "tangential",
    samples_per_space: int = 120,
) -> Path:
    output_path = Path(output_path)
    layout = compute_cavity_layout(cavity)
    points = np.array(layout.beam_points, dtype=float)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    _draw_beam_envelope(ax, cavity, layout.beam_points, plane, samples_per_space)
    if len(points) > 1:
        ax.plot(points[:, 0] * 1e3, points[:, 1] * 1e3, color="#f28e2b", linewidth=2.0, label="beam path")
        ax.scatter(points[:, 0] * 1e3, points[:, 1] * 1e3, s=16, color="#f28e2b", zorder=3)
    _draw_closure_gap(ax, layout)

    symbol_size = _max_symbol_size(layout, cavity.layout.mirror_size)
    for optic in layout.optics:
        _draw_optic(ax, optic, cavity.layout.mirror_size)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Cavity layout from signed AOI")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    _pad_axes(ax, points, symbol_size)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _draw_beam_envelope(
    ax: plt.Axes,
    cavity: Cavity,
    beam_points: list[tuple[float, float]],
    plane: Plane,
    samples_per_space: int,
) -> None:
    envelopes = _beam_envelope_segments(cavity, beam_points, plane, samples_per_space)
    if not envelopes:
        return
    scale = cavity.layout.beam_radius_scale
    label = f"{plane} beam radius"
    if scale != 1.0:
        label += f" x{scale:g}"
    for index, (upper, lower) in enumerate(envelopes):
        polygon = np.vstack([upper, lower[::-1]]) * 1e3
        ax.fill(
            polygon[:, 0],
            polygon[:, 1],
            color="#f28e2b",
            alpha=0.20,
            linewidth=0,
            label=label if index == 0 else "_nolegend_",
        )
        ax.plot(upper[:, 0] * 1e3, upper[:, 1] * 1e3, color="#f28e2b", linewidth=0.8, alpha=0.45, label="_nolegend_")
        ax.plot(lower[:, 0] * 1e3, lower[:, 1] * 1e3, color="#f28e2b", linewidth=0.8, alpha=0.45, label="_nolegend_")


def _draw_closure_gap(ax: plt.Axes, layout: CavityLayout) -> bool:
    if layout.is_closed or len(layout.beam_points) < 2:
        return False
    start = np.array(layout.beam_points[0], dtype=float)
    end = np.array(layout.beam_points[-1], dtype=float)
    gap = np.vstack([end, start]) * 1e3
    ax.plot(gap[:, 0], gap[:, 1], color="#d62728", linestyle="--", linewidth=1.8, label="closure gap")
    ax.scatter(gap[0, 0], gap[0, 1], marker="x", s=48, color="#d62728", zorder=4, label="_nolegend_")
    return True


def _beam_envelope_points(
    cavity: Cavity,
    beam_points: list[tuple[float, float]],
    plane: Plane,
    samples_per_space: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    segments = _beam_envelope_segments(cavity, beam_points, plane, samples_per_space)
    if not segments:
        return None
    return np.vstack([segment[0] for segment in segments]), np.vstack([segment[1] for segment in segments])


def _beam_envelope_segments(
    cavity: Cavity,
    beam_points: list[tuple[float, float]],
    plane: Plane,
    samples_per_space: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    points = np.array(beam_points, dtype=float)
    if len(points) < 2:
        return []
    try:
        samples = cavity.sample_beam(plane, samples_per_space=samples_per_space)
    except ValueError:
        return []

    segment_vectors = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = cumulative[-1]
    if total_length <= 0:
        return []

    uppers: list[list[np.ndarray]] = [[] for _ in segment_lengths]
    lowers: list[list[np.ndarray]] = [[] for _ in segment_lengths]
    scale = cavity.layout.beam_radius_scale
    for sample in samples:
        position = min(max(sample.position, 0.0), total_length)
        index = int(np.searchsorted(cumulative, position, side="right") - 1)
        index = min(max(index, 0), len(segment_lengths) - 1)
        if segment_lengths[index] <= 0:
            continue
        fraction = (position - cumulative[index]) / segment_lengths[index]
        center = points[index] + fraction * segment_vectors[index]
        tangent = segment_vectors[index] / segment_lengths[index]
        normal = np.array([-tangent[1], tangent[0]])
        radius = sample.beam_radius * scale
        uppers[index].append(center + radius * normal)
        lowers[index].append(center - radius * normal)

    envelopes: list[tuple[np.ndarray, np.ndarray]] = []
    for upper, lower in zip(uppers, lowers):
        if len(upper) >= 2:
            envelopes.append((np.array(upper), np.array(lower)))
    return envelopes


def _draw_optic(ax: plt.Axes, optic: LayoutOptic, default_symbol_size: float) -> None:
    position = np.array(optic.position, dtype=float)
    symbol_size = optic.mirror_size if optic.mirror_size is not None else default_symbol_size
    if optic.element_type in {"FlatMirror", "CurvedMirror"}:
        if optic.tangent_angle is None:
            ax.plot(position[0] * 1e3, position[1] * 1e3, "s", color="#4e79a7")
        elif optic.element_type == "CurvedMirror" and optic.radius_of_curvature not in {None, float("inf")}:
            _draw_curved_mirror(ax, position, optic.tangent_angle, optic.radius_of_curvature, symbol_size)
        else:
            _draw_flat_mirror(ax, position, optic.tangent_angle, symbol_size)
    elif optic.element_type == "ThinLens":
        angle = (optic.direction_in or 0.0) + np.pi / 2.0
        _draw_lens(ax, position, angle, symbol_size)
    else:
        ax.plot(position[0] * 1e3, position[1] * 1e3, "o", color="#59a14f")

    offset = np.array([symbol_size * 0.18, symbol_size * 0.18])
    ax.text(
        (position[0] + offset[0]) * 1e3,
        (position[1] + offset[1]) * 1e3,
        _optic_label(optic),
        fontsize=8,
        ha="left",
        va="bottom",
    )


def _draw_flat_mirror(ax: plt.Axes, position: np.ndarray, tangent_angle: float, symbol_size: float) -> None:
    tangent = np.array([cos(tangent_angle), sin(tangent_angle)])
    half = 0.5 * symbol_size * tangent
    segment = np.vstack([position - half, position + half]) * 1e3
    ax.plot(segment[:, 0], segment[:, 1], color="#4e79a7", linewidth=3.0, solid_capstyle="round", label="_nolegend_")


def _draw_curved_mirror(
    ax: plt.Axes,
    position: np.ndarray,
    tangent_angle: float,
    radius_of_curvature: float | None,
    symbol_size: float,
) -> None:
    curve_array = _curved_mirror_points(position, tangent_angle, radius_of_curvature, symbol_size) * 1e3
    sign = 1.0 if (radius_of_curvature or 1.0) > 0 else -1.0
    color = "#4e79a7" if sign > 0 else "#e15759"
    ax.plot(curve_array[:, 0], curve_array[:, 1], color=color, linewidth=3.0, label="_nolegend_")


def _draw_lens(ax: plt.Axes, position: np.ndarray, tangent_angle: float, symbol_size: float) -> None:
    tangent = np.array([cos(tangent_angle), sin(tangent_angle)])
    half = 0.45 * symbol_size * tangent
    segment = np.vstack([position - half, position + half]) * 1e3
    ax.plot(segment[:, 0], segment[:, 1], color="#59a14f", linewidth=2.2, label="_nolegend_")
    ax.plot(position[0] * 1e3, position[1] * 1e3, "o", color="#59a14f", markersize=3)


def _optic_label(optic: LayoutOptic) -> str:
    if optic.element_type == "CurvedMirror" and optic.radius_of_curvature is not None and isfinite(optic.radius_of_curvature):
        return f"{optic.name}\nRc={optic.radius_of_curvature * 1e3:.1f} mm"
    if optic.angle_of_incidence not in {None, 0.0}:
        return f"{optic.name}\nAOI={optic.angle_of_incidence * 180 / np.pi:.1f} deg"
    return optic.name


def _max_symbol_size(layout: CavityLayout, default_symbol_size: float) -> float:
    sizes = [default_symbol_size]
    sizes.extend(optic.mirror_size for optic in layout.optics if optic.mirror_size is not None)
    return max(sizes)


def _curved_mirror_points(
    position: np.ndarray,
    tangent_angle: float,
    radius_of_curvature: float | None,
    mirror_size: float,
) -> np.ndarray:
    tangent = np.array([cos(tangent_angle), sin(tangent_angle)])
    normal = np.array([-sin(tangent_angle), cos(tangent_angle)])
    radius = radius_of_curvature or float("inf")
    if not isfinite(radius):
        half = 0.5 * mirror_size * tangent
        return np.vstack([position - half, position + half])

    abs_radius = abs(radius)
    half_width = min(0.5 * mirror_size, 0.98 * abs_radius)
    s_values = np.linspace(-half_width, half_width, 49)
    sign = 1.0 if radius > 0 else -1.0
    sag_values = radius - sign * np.sqrt(np.maximum(abs_radius**2 - s_values**2, 0.0))
    return np.array([position + s * tangent + sag * normal for s, sag in zip(s_values, sag_values)])


def _pad_axes(ax: plt.Axes, points: np.ndarray, symbol_size: float) -> None:
    if len(points) == 0:
        return
    min_xy = np.min(points, axis=0) * 1e3
    max_xy = np.max(points, axis=0) * 1e3
    pad = max(symbol_size * 1e3 * 2.0, 5.0)
    if abs(max_xy[0] - min_xy[0]) < 1e-9:
        min_xy[0] -= pad
        max_xy[0] += pad
    if abs(max_xy[1] - min_xy[1]) < 1e-9:
        min_xy[1] -= pad
        max_xy[1] += pad
    ax.set_xlim(min_xy[0] - pad, max_xy[0] + pad)
    ax.set_ylim(min_xy[1] - pad, max_xy[1] + pad)
