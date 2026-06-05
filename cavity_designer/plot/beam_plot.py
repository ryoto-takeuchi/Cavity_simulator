"""Beam propagation plots."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cavity_designer.core.cavity import Cavity, samples_to_arrays
from cavity_designer.core.elements import PLANES, Plane


def plot_beam_radius(
    cavity: Cavity,
    output_path: str | Path,
    planes: Iterable[Plane] = PLANES,
    samples_per_space: int = 120,
) -> Path:
    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for plane in planes:
        arrays = samples_to_arrays(cavity.sample_beam(plane, samples_per_space=samples_per_space))
        ax.plot(arrays["position"], arrays["beam_radius"] * 1e6, label=plane)
    _mark_elements(ax, cavity)
    ax.set_xlabel("Position along round trip (m)")
    ax.set_ylabel("Beam radius w (um)")
    ax.set_title("Gaussian beam radius")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_wavefront_curvature(
    cavity: Cavity,
    output_path: str | Path,
    planes: Iterable[Plane] = PLANES,
    samples_per_space: int = 120,
) -> Path:
    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for plane in planes:
        arrays = samples_to_arrays(cavity.sample_beam(plane, samples_per_space=samples_per_space))
        curvature = arrays["wavefront_curvature"].copy()
        curvature[~np.isfinite(curvature)] = np.nan
        ax.plot(arrays["position"], curvature, label=plane)
    _mark_elements(ax, cavity)
    ax.set_xlabel("Position along round trip (m)")
    ax.set_ylabel("Wavefront curvature radius R (m)")
    ax.set_title("Wavefront curvature")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _mark_elements(ax: plt.Axes, cavity: Cavity) -> None:
    ylim = ax.get_ylim()
    for marker in cavity.element_markers():
        ax.axvline(marker.position, color="0.55", linewidth=0.8, alpha=0.5)
        ax.text(
            marker.position,
            ylim[1],
            marker.name,
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
            color="0.25",
        )
