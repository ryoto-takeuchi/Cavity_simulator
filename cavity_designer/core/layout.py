"""Two-dimensional cavity layout reconstruction for plotting."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import atan2, cos, pi, sin

import numpy as np

from cavity_designer.core.cavity import Cavity, CavityError, LayoutClosureAdjustment
from cavity_designer.core.constants import ABS_TOL
from cavity_designer.core.elements import Mirror, OpticalElement, Space

LAYOUT_CLOSURE_POSITION_TOL = 1e-9
LAYOUT_CLOSURE_DIRECTION_TOL = 1e-9


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
    position_tolerance: float = LAYOUT_CLOSURE_POSITION_TOL
    direction_tolerance: float = LAYOUT_CLOSURE_DIRECTION_TOL

    @property
    def is_closed(self) -> bool:
        return self.closure_error <= self.position_tolerance and self.direction_error <= self.direction_tolerance


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
        position_tolerance=cavity.layout.closure.position_tolerance,
        direction_tolerance=cavity.layout.closure.direction_tolerance,
    )


def apply_layout_closure_adjustment(cavity: Cavity) -> Cavity:
    """Adjust declared space lengths so the reconstructed layout closes."""

    closure = cavity.layout.closure
    if not closure.enabled:
        return cavity
    if not closure.variables:
        raise CavityError("layout.closure.enabled requires at least one variable.")

    variable_names = _parse_length_variables(cavity, closure.variables)
    initial_layout = compute_cavity_layout(cavity)
    if initial_layout.direction_error > closure.direction_tolerance:
        raise CavityError(
            "Layout closure adjustment cannot fix direction error using space lengths only. "
            f"Direction error is {initial_layout.direction_error:.9g} rad."
        )

    if initial_layout.closure_error <= closure.position_tolerance:
        lengths = {name: _space_by_name(cavity, name).length for name in variable_names}
        return _cavity_with_adjusted_spaces(cavity, lengths)

    sensitivity = _length_sensitivity_matrix(cavity, variable_names)
    target = -np.array(
        [
            initial_layout.beam_points[-1][0] - initial_layout.beam_points[0][0],
            initial_layout.beam_points[-1][1] - initial_layout.beam_points[0][1],
        ],
        dtype=float,
    )
    delta, *_ = np.linalg.lstsq(sensitivity, target, rcond=None)

    adjusted_lengths: dict[str, float] = {}
    for name, change in zip(variable_names, delta):
        original = _space_by_name(cavity, name).length
        adjusted = original + float(change)
        if adjusted < -ABS_TOL:
            raise CavityError(
                f"Layout closure adjustment would make {name}.length negative ({adjusted * 1e3:.9g} mm)."
            )
        adjusted_lengths[name] = max(0.0, adjusted)

    adjusted_cavity = _cavity_with_adjusted_spaces(cavity, adjusted_lengths)
    adjusted_layout = compute_cavity_layout(adjusted_cavity)
    if adjusted_layout.closure_error > closure.position_tolerance:
        raise CavityError(
            "Layout closure adjustment did not close the path with the declared length variables. "
            f"Residual position error is {adjusted_layout.closure_error * 1e3:.9g} mm."
        )
    return adjusted_cavity


def _parse_length_variables(cavity: Cavity, variables: tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    names: list[str] = []
    for variable in variables:
        if not variable.endswith(".length"):
            raise CavityError(
                f"Layout closure variable '{variable}' is not supported; "
                "only '<space_name>.length' variables are supported."
            )
        name = variable[: -len(".length")]
        if not name:
            raise CavityError("Layout closure variable must name a space, such as 'arm_41.length'.")
        if name in seen:
            raise CavityError(f"Layout closure variable '{variable}' is listed more than once.")
        _space_by_name(cavity, name)
        seen.add(name)
        names.append(name)
    return names


def _space_by_name(cavity: Cavity, name: str) -> Space:
    matches = [element for element in cavity.elements if element.name == name]
    if len(matches) != 1:
        if not matches:
            raise CavityError(f"Layout closure variable refers to unknown element '{name}'.")
        raise CavityError(f"Layout closure variable refers to non-unique element name '{name}'.")
    element = matches[0]
    if not isinstance(element, Space):
        raise CavityError(f"Layout closure variable '{name}.length' must refer to a space element.")
    return element


def _length_sensitivity_matrix(cavity: Cavity, variable_names: list[str]) -> np.ndarray:
    direction = cavity.layout.direction
    tangents_by_name: dict[str, float] = {}
    sensitivity = np.zeros((2, len(variable_names)), dtype=float)
    variable_index = {name: index for index, name in enumerate(variable_names)}

    for element in cavity.sequence_for_round_trip():
        if isinstance(element, Space):
            index = variable_index.get(element.name)
            if index is not None:
                sensitivity[:, index] += _unit(direction)
            continue

        if isinstance(element, Mirror):
            tangent = tangents_by_name.get(element.name)
            if tangent is None:
                tangent = direction + pi / 2.0 - getattr(element, "angle_of_incidence", 0.0)
                tangents_by_name[element.name] = tangent
            direction = _reflect_direction(direction, tangent)

    if np.linalg.norm(sensitivity, ord=2) <= ABS_TOL:
        raise CavityError("Layout closure variables do not affect the reconstructed path position.")
    return sensitivity


def _cavity_with_adjusted_spaces(cavity: Cavity, adjusted_lengths: dict[str, float]) -> Cavity:
    adjustments: list[LayoutClosureAdjustment] = []
    new_elements: list[OpticalElement] = []
    for element in cavity.elements:
        if isinstance(element, Space) and element.name in adjusted_lengths:
            adjusted_length = adjusted_lengths[element.name]
            adjustments.append(
                LayoutClosureAdjustment(
                    variable=f"{element.name}.length",
                    original_length=element.length,
                    adjusted_length=adjusted_length,
                )
            )
            new_elements.append(replace(element, length_m=adjusted_length))
        else:
            new_elements.append(element)

    return Cavity(
        cavity_type=cavity.cavity_type,
        wavelength=cavity.wavelength,
        elements=new_elements,
        layout=cavity.layout,
        layout_adjustments=tuple(adjustments),
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
