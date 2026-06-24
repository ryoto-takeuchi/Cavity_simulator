"""YAML configuration parser with explicit unit handling."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from cavity_designer.core.cavity import Cavity, LayoutClosureConfig, LayoutConfig
from cavity_designer.core.elements import CurvedMirror, CustomABCD, FlatMirror, Space, ThinLens
from cavity_designer.core.layout import apply_layout_closure_adjustment


class ConfigError(ValueError):
    """Raised when a YAML cavity configuration is invalid."""


_NUMBER_UNIT_RE = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-z%µ]+)?\s*$"
)

_LENGTH_UNITS = {
    "nm": 1e-9,
    "um": 1e-6,
    "µm": 1e-6,
    "mm": 1e-3,
    "cm": 1e-2,
    "m": 1.0,
}

_POWER_UNITS = {
    "ppm": 1e-6,
    "%": 1e-2,
}

_ANGLE_UNITS = {
    "rad": 1.0,
    "mrad": 1e-3,
    "deg": 3.141592653589793 / 180.0,
}


def load_cavity(path: str | Path) -> Cavity:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return parse_cavity_config(raw)


def parse_cavity_config(raw: dict[str, Any]) -> Cavity:
    if not isinstance(raw, dict):
        raise ConfigError("Configuration root must be a mapping.")
    if "wavelength" not in raw:
        raise ConfigError("Missing required field 'wavelength'.")
    wavelength = parse_length(raw["wavelength"], "wavelength")

    cavity_raw = raw.get("cavity", {})
    if not isinstance(cavity_raw, dict):
        raise ConfigError("'cavity' must be a mapping.")
    cavity_type = str(cavity_raw.get("type", "linear")).lower()
    layout = _parse_layout(raw.get("layout"))

    elements_raw = raw.get("elements")
    if not isinstance(elements_raw, list) or not elements_raw:
        raise ConfigError("Configuration must contain a non-empty 'elements' list.")
    elements = [_parse_element(index, element_raw) for index, element_raw in enumerate(elements_raw, start=1)]
    cavity = Cavity(cavity_type=cavity_type, wavelength=wavelength, elements=elements, layout=layout)
    return apply_layout_closure_adjustment(cavity)


def parse_length(value: Any, field_name: str, allow_inf: bool = False, allow_negative: bool = False) -> float:
    if allow_inf and isinstance(value, str) and value.strip().lower() in {"inf", "+inf", "infinity", "+infinity"}:
        return float("inf")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        raise ConfigError(f"Length value '{value}' requires an explicit unit.")
    number, unit = _parse_number_unit(value, field_name)
    if unit not in _LENGTH_UNITS:
        raise ConfigError(f"Invalid length unit '{unit}' for {field_name}.")
    length = number * _LENGTH_UNITS[unit]
    if length < 0 and not allow_negative:
        raise ConfigError(f"{field_name} must be non-negative.")
    return length


def parse_power(value: Any, field_name: str, default: float | None = None) -> float:
    if value is None:
        if default is None:
            raise ConfigError(f"Missing required power quantity '{field_name}'.")
        return default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        parsed = float(value)
    else:
        number, unit = _parse_number_unit(value, field_name)
        if unit is None:
            parsed = number
        elif unit in _POWER_UNITS:
            parsed = number * _POWER_UNITS[unit]
        else:
            raise ConfigError(f"Invalid power unit '{unit}' for {field_name}.")
    if parsed < 0:
        raise ConfigError(f"{field_name} must be non-negative.")
    return parsed


def parse_dimensionless(value: Any, field_name: str, default: float | None = None) -> float:
    if value is None:
        if default is None:
            raise ConfigError(f"Missing required dimensionless value '{field_name}'.")
        return default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    number, unit = _parse_number_unit(value, field_name)
    if unit is not None:
        raise ConfigError(f"{field_name} must be dimensionless.")
    return number


def parse_angle(value: Any, field_name: str, default: float = 0.0, allow_negative: bool = False) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        raise ConfigError(f"Angle value '{value}' requires an explicit unit.")
    number, unit = _parse_number_unit(value, field_name)
    if unit not in _ANGLE_UNITS:
        raise ConfigError(f"Invalid angle unit '{unit}' for {field_name}.")
    angle = number * _ANGLE_UNITS[unit]
    if angle < 0 and not allow_negative:
        raise ConfigError(f"{field_name} must be non-negative.")
    return angle


def _parse_element(index: int, raw: Any) -> object:
    if not isinstance(raw, dict):
        raise ConfigError(f"Element {index} must be a mapping.")
    name = str(raw.get("name", f"element_{index}"))
    element_type = str(raw.get("type", "")).lower()
    if not element_type:
        raise ConfigError(f"Element {name} is missing required field 'type'.")

    if element_type == "space":
        if "length" not in raw:
            raise ConfigError(f"Space {name} is missing required field 'length'.")
        return Space(name=name, length_m=parse_length(raw["length"], f"{name}.length"))

    if element_type in {"thin_lens", "lens"}:
        focal_value = raw.get("f", raw.get("focal_length"))
        power_loss = parse_power(raw.get("L", raw.get("power_loss")), f"{name}.L", 0.0)
        if focal_value is not None:
            return ThinLens(
                name=name,
                focal_length=parse_length(focal_value, f"{name}.f", allow_negative=True),
                power_loss_value=power_loss,
            )
        tangential_value = raw.get("f_tangential", raw.get("f_t", raw.get("ft")))
        sagittal_value = raw.get("f_sagittal", raw.get("f_s", raw.get("fs")))
        if tangential_value is None or sagittal_value is None:
            raise ConfigError(f"Thin lens {name} requires 'f' or both 'f_tangential' and 'f_sagittal'.")
        return ThinLens(
            name=name,
            focal_length_tangential=parse_length(tangential_value, f"{name}.f_tangential", allow_negative=True),
            focal_length_sagittal=parse_length(sagittal_value, f"{name}.f_sagittal", allow_negative=True),
            power_loss_value=power_loss,
        )

    if element_type in {"mirror", "flat_mirror"}:
        return FlatMirror(
            name=name,
            power_transmission_value=parse_power(raw.get("T", raw.get("power_transmission")), f"{name}.T", 0.0),
            power_loss_value=parse_power(raw.get("L", raw.get("power_loss")), f"{name}.L", 0.0),
            power_reflectivity=_optional_reflectivity(raw, name),
            layout_mirror_size=_optional_element_layout_mirror_size(raw, name),
            angle_of_incidence=parse_angle(
                raw.get("AOI", raw.get("angle_of_incidence")), f"{name}.AOI", allow_negative=True
            ),
        )

    if element_type in {"curved_mirror", "spherical_mirror"}:
        radius_value = raw.get("Rc", raw.get("radius_of_curvature"))
        if radius_value is None:
            raise ConfigError(f"Curved mirror {name} is missing required field 'Rc'.")
        return CurvedMirror(
            name=name,
            power_transmission_value=parse_power(raw.get("T", raw.get("power_transmission")), f"{name}.T", 0.0),
            power_loss_value=parse_power(raw.get("L", raw.get("power_loss")), f"{name}.L", 0.0),
            power_reflectivity=_optional_reflectivity(raw, name),
            layout_mirror_size=_optional_element_layout_mirror_size(raw, name),
            radius_of_curvature=parse_length(radius_value, f"{name}.Rc", allow_inf=True, allow_negative=True),
            angle_of_incidence=parse_angle(
                raw.get("AOI", raw.get("angle_of_incidence")), f"{name}.AOI", allow_negative=True
            ),
        )

    if element_type in {"custom_abcd", "abcd"}:
        missing = [key for key in ("A", "B", "C", "D") if key not in raw]
        if missing:
            raise ConfigError(f"Custom ABCD {name} is missing required fields: {', '.join(missing)}.")
        return CustomABCD(name=name, A=float(raw["A"]), B=float(raw["B"]), C=float(raw["C"]), D=float(raw["D"]))

    raise ConfigError(f"Unknown element type '{element_type}' for element {name}.")


def _parse_layout(raw: Any) -> LayoutConfig:
    if raw is None:
        return LayoutConfig()
    if not isinstance(raw, dict):
        raise ConfigError("'layout' must be a mapping.")

    if "start" in raw:
        start_raw = raw["start"]
        if not isinstance(start_raw, (list, tuple)) or len(start_raw) != 2:
            raise ConfigError("layout.start must be a two-element list such as ['0 mm', '0 mm'].")
        start = (
            parse_length(start_raw[0], "layout.start[0]", allow_negative=True),
            parse_length(start_raw[1], "layout.start[1]", allow_negative=True),
        )
    else:
        start = (
            parse_length(raw["start_x"], "layout.start_x", allow_negative=True) if "start_x" in raw else 0.0,
            parse_length(raw["start_y"], "layout.start_y", allow_negative=True) if "start_y" in raw else 0.0,
        )

    direction = parse_angle(raw.get("direction"), "layout.direction", allow_negative=True)
    mirror_size = parse_length(raw.get("mirror_size", "10 mm"), "layout.mirror_size")
    if mirror_size <= 0:
        raise ConfigError("layout.mirror_size must be positive.")
    beam_radius_scale = parse_dimensionless(raw.get("beam_radius_scale"), "layout.beam_radius_scale", 1.0)
    if beam_radius_scale <= 0:
        raise ConfigError("layout.beam_radius_scale must be positive.")
    closure = _parse_layout_closure(raw.get("closure"))
    return LayoutConfig(
        start=start,
        direction=direction,
        mirror_size=mirror_size,
        beam_radius_scale=beam_radius_scale,
        closure=closure,
    )


def _parse_layout_closure(raw: Any) -> LayoutClosureConfig:
    if raw is None:
        return LayoutClosureConfig()
    if not isinstance(raw, dict):
        raise ConfigError("layout.closure must be a mapping.")

    variables_raw = raw.get("variables", ())
    if variables_raw is None:
        variables_raw = ()
    if not isinstance(variables_raw, (list, tuple)):
        raise ConfigError("layout.closure.variables must be a list of strings.")
    variables = tuple(str(variable) for variable in variables_raw)
    enabled = bool(raw.get("enabled", bool(variables)))
    if enabled and not variables:
        raise ConfigError("layout.closure.enabled requires layout.closure.variables.")

    position_tolerance = parse_length(
        raw.get("position_tolerance", "1 nm"),
        "layout.closure.position_tolerance",
    )
    direction_tolerance = parse_angle(
        raw.get("direction_tolerance", "1e-9 rad"),
        "layout.closure.direction_tolerance",
    )
    return LayoutClosureConfig(
        enabled=enabled,
        variables=variables,
        position_tolerance=position_tolerance,
        direction_tolerance=direction_tolerance,
    )


def _optional_reflectivity(raw: dict[str, Any], name: str) -> float | None:
    value = raw.get("R", raw.get("power_reflectivity"))
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"inf", "+inf", "infinity", "+infinity"}:
        return None
    return parse_power(value, f"{name}.R")


def _optional_element_layout_mirror_size(raw: dict[str, Any], name: str) -> float | None:
    layout_raw = raw.get("layout")
    if layout_raw is None:
        return None
    if not isinstance(layout_raw, dict):
        raise ConfigError(f"{name}.layout must be a mapping.")
    if "mirror_size" not in layout_raw:
        return None
    mirror_size = parse_length(layout_raw["mirror_size"], f"{name}.layout.mirror_size")
    if mirror_size <= 0:
        raise ConfigError(f"{name}.layout.mirror_size must be positive.")
    return mirror_size


def _parse_number_unit(value: Any, field_name: str) -> tuple[float, str | None]:
    if not isinstance(value, str):
        raise ConfigError(f"Value for {field_name} must be a string with an explicit unit.")
    match = _NUMBER_UNIT_RE.match(value)
    if not match:
        raise ConfigError(f"Could not parse value '{value}' for {field_name}.")
    number = float(match.group(1))
    unit = match.group(2)
    return number, unit
