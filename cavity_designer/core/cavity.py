"""Cavity-level ABCD, stability, and beam propagation."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Iterable

import numpy as np

from cavity_designer.core.constants import ABS_TOL
from cavity_designer.core.elements import PLANES, Mirror, OpticalElement, Plane, Space
from cavity_designer.core.gaussian import (
    GaussianMode,
    beam_radius_from_q,
    gouy_phase_from_q,
    propagate_q,
    round_trip_gouy_phase,
    solve_eigenmode,
)
from cavity_designer.core.matrix import ABCDMatrix, compose


class CavityError(ValueError):
    """Raised for invalid cavity-level operations."""


@dataclass(frozen=True)
class StabilityResult:
    plane: Plane
    matrix: ABCDMatrix
    trace_half: float
    stable: bool


@dataclass(frozen=True)
class BeamSample:
    plane: Plane
    position: float
    q: complex
    beam_radius: float
    wavefront_curvature: float
    gouy_phase: float


@dataclass(frozen=True)
class ElementMarker:
    position: float
    name: str
    element_type: str


@dataclass(frozen=True)
class LayoutClosureConfig:
    enabled: bool = False
    variables: tuple[str, ...] = ()
    position_tolerance: float = 1e-9
    direction_tolerance: float = 1e-9


@dataclass(frozen=True)
class LayoutClosureAdjustment:
    variable: str
    original_length: float
    adjusted_length: float

    @property
    def delta(self) -> float:
        return self.adjusted_length - self.original_length


@dataclass(frozen=True)
class LayoutConfig:
    start: tuple[float, float] = (0.0, 0.0)
    direction: float = 0.0
    mirror_size: float = 0.01
    beam_radius_scale: float = 1.0
    closure: LayoutClosureConfig = LayoutClosureConfig()


@dataclass
class Cavity:
    cavity_type: str
    wavelength: float
    elements: list[OpticalElement]
    layout: LayoutConfig = LayoutConfig()
    layout_adjustments: tuple[LayoutClosureAdjustment, ...] = ()

    def __post_init__(self) -> None:
        if self.cavity_type not in {"linear", "ring"}:
            raise CavityError("Cavity type must be 'linear' or 'ring'.")
        if self.wavelength <= 0:
            raise CavityError("Wavelength must be positive.")
        if not self.elements:
            raise CavityError("Cavity must contain at least one element.")

    def sequence_for_round_trip(self) -> list[OpticalElement]:
        if self.cavity_type == "ring":
            return self.elements[1:] + self.elements[:1]
        if len(self.mirrors) < 2:
            raise CavityError("A linear cavity requires at least two mirrors.")
        return self.elements[1:] + list(reversed(self.elements[:-1]))

    def round_trip_matrix(self, plane: Plane = "tangential") -> ABCDMatrix:
        return compose(element.matrix(plane) for element in self.sequence_for_round_trip())

    def stability(self, plane: Plane = "tangential") -> StabilityResult:
        matrix = self.round_trip_matrix(plane)
        trace_half = matrix.trace_half
        return StabilityResult(
            plane=plane,
            matrix=matrix,
            trace_half=trace_half,
            stable=abs(trace_half) < 1.0 - ABS_TOL,
        )

    def eigenmode(self, plane: Plane = "tangential") -> GaussianMode | None:
        return solve_eigenmode(self.round_trip_matrix(plane), self.wavelength)

    def round_trip_gouy_phase(self, plane: Plane = "tangential") -> float | None:
        return round_trip_gouy_phase(self.round_trip_matrix(plane))

    def eigenmodes(self) -> dict[Plane, GaussianMode | None]:
        return {plane: self.eigenmode(plane) for plane in PLANES}

    @property
    def mirrors(self) -> list[Mirror]:
        return [element for element in self.elements if isinstance(element, Mirror)]

    @property
    def one_way_path_length(self) -> float:
        return sum(element.length for element in self.elements)

    @property
    def round_trip_length(self) -> float:
        if self.cavity_type == "linear":
            return 2.0 * self.one_way_path_length
        return self.one_way_path_length

    def optical_path_length(self) -> float:
        return self.round_trip_length

    def is_simple_two_mirror_linear(self) -> bool:
        return (
            self.cavity_type == "linear"
            and len(self.mirrors) == 2
            and all(isinstance(element, (Mirror, Space)) for element in self.elements)
        )

    def two_mirror_g_parameters(self) -> tuple[float, float] | None:
        if not self.is_simple_two_mirror_linear():
            return None
        mirrors = self.mirrors
        radii: list[float] = []
        for mirror in mirrors:
            radius = getattr(mirror, "radius_of_curvature", float("inf"))
            radii.append(radius)
        length = self.one_way_path_length
        g_values = tuple(1.0 - length / radius if np.isfinite(radius) else 1.0 for radius in radii)
        return g_values  # type: ignore[return-value]

    def element_markers(self) -> list[ElementMarker]:
        markers: list[ElementMarker] = []
        position = 0.0
        first = self.elements[0]
        if not isinstance(first, Space):
            markers.append(ElementMarker(position=0.0, name=first.name, element_type=type(first).__name__))
        for element in self.sequence_for_round_trip():
            if isinstance(element, Space):
                position += element.length
            else:
                markers.append(ElementMarker(position=position, name=element.name, element_type=type(element).__name__))
        return markers

    def sample_beam(self, plane: Plane = "tangential", samples_per_space: int = 80) -> list[BeamSample]:
        mode = self.eigenmode(plane)
        if mode is None:
            raise CavityError("No stable Gaussian eigenmode exists for this cavity.")
        samples: list[BeamSample] = []
        position = 0.0
        q_current = mode.q
        _append_sample(samples, plane, position, q_current, self.wavelength)

        for element in self.sequence_for_round_trip():
            if isinstance(element, Space):
                steps = max(2, int(ceil(samples_per_space * max(element.length / max(self.round_trip_length, ABS_TOL), 0.05))))
                start_q = q_current
                for index in range(1, steps + 1):
                    fraction = index / steps
                    q_in_space = start_q + element.length * fraction
                    _append_sample(samples, plane, position + element.length * fraction, q_in_space, self.wavelength)
                position += element.length
                q_current = start_q + element.length
            else:
                q_current = propagate_q(q_current, element.matrix(plane))
                _append_sample(samples, plane, position, q_current, self.wavelength)

        return samples

    def round_trip_power_loss(self) -> float:
        loss = sum(element.power_transmission + element.power_loss for element in self.sequence_for_round_trip())
        if self.cavity_type == "ring" and self.elements[0].is_mirror:
            # sequence_for_round_trip includes the first mirror once at the end.
            return loss
        return loss


def _append_sample(samples: list[BeamSample], plane: Plane, position: float, q: complex, wavelength: float) -> None:
    samples.append(
        BeamSample(
            plane=plane,
            position=position,
            q=q,
            beam_radius=beam_radius_from_q(q, wavelength),
            wavefront_curvature=_curvature(q),
            gouy_phase=gouy_phase_from_q(q),
        )
    )


def _curvature(q: complex) -> float:
    real_part = (1.0 / q).real
    if abs(real_part) < ABS_TOL:
        return float("inf")
    return 1.0 / real_part


def samples_to_arrays(samples: Iterable[BeamSample]) -> dict[str, np.ndarray]:
    sample_list = list(samples)
    return {
        "position": np.array([sample.position for sample in sample_list]),
        "beam_radius": np.array([sample.beam_radius for sample in sample_list]),
        "wavefront_curvature": np.array([sample.wavefront_curvature for sample in sample_list]),
        "gouy_phase": np.array([sample.gouy_phase for sample in sample_list]),
        "q": np.array([sample.q for sample in sample_list], dtype=complex),
    }
