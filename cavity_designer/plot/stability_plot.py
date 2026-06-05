"""Stability plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cavity_designer.core.matrix import ABCDMatrix


def plot_g1_g2(output_path: str | Path, g1: float | None = None, g2: float | None = None) -> Path:
    output_path = Path(output_path)
    values = np.linspace(-1.5, 1.5, 400)
    G1, G2 = np.meshgrid(values, values)
    stable = (G1 * G2 > 0.0) & (G1 * G2 < 1.0)

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.contourf(G1, G2, stable.astype(float), levels=[-0.5, 0.5, 1.5], colors=["white", "#cfe8ff"], alpha=0.9)
    ax.contour(G1, G2, G1 * G2, levels=[0.0, 1.0], colors=["0.2"], linewidths=1.0)
    if g1 is not None and g2 is not None:
        ax.plot([g1], [g2], "o", color="#b00020", label="cavity")
        ax.legend()
    ax.set_xlabel("g1")
    ax.set_ylabel("g2")
    ax.set_title("Two-mirror stability plane")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_stability_map(
    x_values: np.ndarray,
    y_values: np.ndarray,
    matrix_factory: Callable[[float, float], ABCDMatrix],
    output_path: str | Path,
    x_label: str = "x",
    y_label: str = "y",
) -> Path:
    output_path = Path(output_path)
    stable = np.zeros((len(y_values), len(x_values)), dtype=float)
    trace_half = np.zeros_like(stable)
    for yi, y_value in enumerate(y_values):
        for xi, x_value in enumerate(x_values):
            matrix = matrix_factory(float(x_value), float(y_value))
            trace_half[yi, xi] = matrix.trace_half
            stable[yi, xi] = 1.0 if abs(matrix.trace_half) < 1.0 else 0.0

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    mesh = ax.pcolormesh(x_values, y_values, stable, shading="auto", cmap="Blues", vmin=0, vmax=1)
    ax.contour(x_values, y_values, np.abs(trace_half), levels=[1.0], colors=["0.15"], linewidths=1.0)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title("ABCD stability map")
    fig.colorbar(mesh, ax=ax, label="Stable")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path
