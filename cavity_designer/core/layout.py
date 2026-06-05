"""Two-dimensional cavity layout reconstruction for plotting."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, pi, sin

import numpy as np

from cavity_designer.core.cavity import Cavity
from cavity_designer.core.elements import Mirror, OpticalElement, Space


@dataclass(frozen=True)
class LayoutOptic:
    name: str
    element_type: str
    position: tuple[float, float]
    direction_in: float | None = None
    direction_out: float | None = None
    tangent_angle: float | None = None
    angle_of_incidence: float | None = None
    radius_of_curvature: float | None = None


@dataclass(frozen=True)
class CavityLayout:
    beam_points: list[tuple[float, float]]
    optics: list[LayoutOptic]
    closure_error: float
    direction_error: float


def compute_cavity_layout(cavity: Cavity) -> CavityLayout:
    """Reconstruct a 2D layout from spaces and signed AOIs.

    The reference plane is immediately after the first element. The first
    element is placed at layout.start, and layout.direction is the beam direction
    just after that element. At the first encounter with a mirror, signed AOI
    sets the mirror tangent. Repeated encounters reuse the same tangent.
    """

    start = np.array(cavity.layout.start, dtype=float)
    position = start.copy()
    direction = cavity.layout.direction
    initial_direction = direction

    points: list[tuple[float, float]] = [_as_point(position)]
    optics_by_name: dict[str, LayoutOptic] = {}
    tangents_by_name: dict[str, float] = {}

    first = cavity.elements[0]
    _record_optic(optics_by_name, first, position, None, direction, None)

    for element in cavity.sequence_for_round_trip():
        if isinstance(element, Space):
            position = position + element.length * _unit(direction)
            points.append(_as_point(position))
            continue

        direction_in = direction
        tangent = None
        if isinstance(element, Mirror):
            tangent = tangents_by_name.get(element.name)
            if tangent is None:
                aoi = getattr(element, "angle_of_incidence", 0.0)
                tangent = direction + pi / 2.0 - aoi
                tangents_by_name[element.name] = tangent
            direction = _reflect_direction(direction, tangent)

        _record_optic(optics_by_name, element, position, direction_in, direction, tangent)

    closure_error = float(np.linalg.norm(position - start))
    direction_error = abs(_angle_difference(direction, initial_direction))
    return CavityLayout(
        beam_points=points,
        optics=list(optics_by_name.values()),
        closure_error=closure_error,
        direction_error=direction_error,
    )


def _record_optic(
    optics_by_name: dict[str, LayoutOptic],
    element: OpticalElement,
    position: np.ndarray,
    direction_in: float | None,
    direction_out: float | None,
    tangent_angle: float | None,
) -> None:
    existing = optics_by_name.get(element.name)
    optic = LayoutOptic(
        name=element.name,
        element_type=type(element).__name__,
        position=_as_point(position),
        direction_in=direction_in if direction_in is not None else (existing.direction_in if existing else None),
        direction_out=direction_out if direction_out is not None else (existing.direction_out if existing else None),
        tangent_angle=tangent_angle if tangent_angle is not None else (existing.tangent_angle if existing else None),
        angle_of_incidence=getattr(element, "angle_of_incidence", None),
        radius_of_curvature=getattr(element, "radius_of_curvature", None),
    )
    optics_by_name[element.name] = optic


def _unit(angle: float) -> np.ndarray:
    return np.array([cos(angle), sin(angle)], dtype=float)


def _reflect_direction(direction: float, tangent: float) -> float:
    return _normalize_angle(2.0 * tangent - direction)


def _normalize_angle(angle: float) -> float:
    return atan2(sin(angle), cos(angle))


def _angle_difference(a: float, b: float) -> float:
    return _normalize_angle(a - b)


def _as_point(position: np.ndarray) -> tuple[float, float]:
    return (float(position[0]), float(position[1]))
