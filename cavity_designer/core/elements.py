"""Optical element models and plane-aware ABCD matrices."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite
from typing import Literal

from cavity_designer.core.constants import ABS_TOL, REL_TOL
from cavity_designer.core.matrix import (
    ABCDMatrix,
    curved_mirror_reflection_matrix,
    custom_matrix,
    flat_mirror_matrix,
    space_matrix,
    thin_lens_matrix,
)

Plane = Literal["tangential", "sagittal"]
PLANES: tuple[Plane, Plane] = ("tangential", "sagittal")


class ElementError(ValueError):
    """Raised when an optical element is physically invalid."""


@dataclass(frozen=True)
class OpticalElement:
    name: str

    def matrix(self, plane: Plane = "tangential") -> ABCDMatrix:
        raise NotImplementedError

    @property
    def length(self) -> float:
        return 0.0

    @property
    def power_transmission(self) -> float:
        return 0.0

    @property
    def power_loss(self) -> float:
        return 0.0

    @property
    def is_mirror(self) -> bool:
        return False


@dataclass(frozen=True)
class Space(OpticalElement):
    length_m: float

    def __post_init__(self) -> None:
        if self.length_m < 0:
            raise ElementError(f"Space {self.name} has negative length.")

    @property
    def length(self) -> float:
        return self.length_m

    def matrix(self, plane: Plane = "tangential") -> ABCDMatrix:
        _validate_plane(plane)
        return space_matrix(self.length_m)


@dataclass(frozen=True)
class ThinLens(OpticalElement):
    """Thin lens.

    A scalar focal_length applies to both transverse planes. Plane-specific
    focal lengths allow simple astigmatic or cylindrical-lens models.
    """

    focal_length: float | None = None
    focal_length_tangential: float | None = None
    focal_length_sagittal: float | None = None
    power_loss_value: float = 0.0

    def __post_init__(self) -> None:
        if self.focal_length is None and (
            self.focal_length_tangential is None or self.focal_length_sagittal is None
        ):
            raise ElementError(
                f"Thin lens {self.name} requires either focal_length or both plane-specific focal lengths."
            )
        for label, value in (
            ("focal length", self.focal_length),
            ("tangential focal length", self.focal_length_tangential),
            ("sagittal focal length", self.focal_length_sagittal),
        ):
            if value is not None and abs(value) < ABS_TOL:
                raise ElementError(f"Thin lens {self.name} has zero {label}.")
        if self.power_loss_value < -ABS_TOL:
            raise ElementError(f"Thin lens {self.name} has negative loss.")
        if self.power_loss_value > 1.0 + ABS_TOL:
            raise ElementError(f"Thin lens {self.name} has loss greater than 1.")

    def matrix(self, plane: Plane = "tangential") -> ABCDMatrix:
        _validate_plane(plane)
        return thin_lens_matrix(self.focal_length_for_plane(plane))

    @property
    def power_loss(self) -> float:
        return self.power_loss_value

    def focal_length_for_plane(self, plane: Plane = "tangential") -> float:
        _validate_plane(plane)
        if plane == "tangential" and self.focal_length_tangential is not None:
            return self.focal_length_tangential
        if plane == "sagittal" and self.focal_length_sagittal is not None:
            return self.focal_length_sagittal
        if self.focal_length is None:
            raise ElementError(f"Thin lens {self.name} has no focal length for {plane} plane.")
        return self.focal_length


@dataclass(frozen=True)
class Mirror(OpticalElement):
    power_transmission_value: float = 0.0
    power_loss_value: float = 0.0
    power_reflectivity: float | None = None
    layout_mirror_size: float | None = None

    def __post_init__(self) -> None:
        reflectivity = 1.0 - self.power_transmission_value - self.power_loss_value
        if self.power_reflectivity is not None:
            reflectivity = self.power_reflectivity
        _validate_power_coefficients(
            self.name,
            reflectivity,
            self.power_transmission_value,
            self.power_loss_value,
        )
        object.__setattr__(self, "power_reflectivity", reflectivity)
        if self.layout_mirror_size is not None and self.layout_mirror_size <= 0.0:
            raise ElementError(f"Mirror {self.name} has non-positive layout mirror size.")

    @property
    def power_transmission(self) -> float:
        return self.power_transmission_value

    @property
    def power_loss(self) -> float:
        return self.power_loss_value

    @property
    def is_mirror(self) -> bool:
        return True


@dataclass(frozen=True)
class FlatMirror(Mirror):
    """Plane mirror. Its reflection matrix is identity in this convention."""

    angle_of_incidence: float = 0.0

    def matrix(self, plane: Plane = "tangential") -> ABCDMatrix:
        _validate_plane(plane)
        return flat_mirror_matrix()


@dataclass(frozen=True)
class CurvedMirror(Mirror):
    """Spherical mirror reflection.

    The normal-incidence matrix is [[1, 0], [-2/Rc, 1]]. For oblique
    incidence, tangential and sagittal effective radii are split using
    R_t = Rc cos(theta), R_s = Rc / cos(theta).
    """

    radius_of_curvature: float = float("inf")
    angle_of_incidence: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isfinite(self.radius_of_curvature) and self.radius_of_curvature != float("inf"):
            raise ElementError(f"Curved mirror {self.name} has invalid radius of curvature.")
        if abs(self.radius_of_curvature) < ABS_TOL:
            raise ElementError(f"Curved mirror {self.name} has zero radius of curvature.")
        cos_angle = cos(self.angle_of_incidence)
        if abs(cos_angle) < ABS_TOL:
            raise ElementError(f"Curved mirror {self.name} has grazing incidence.")

    def matrix(self, plane: Plane = "tangential") -> ABCDMatrix:
        _validate_plane(plane)
        if self.radius_of_curvature == float("inf"):
            return flat_mirror_matrix()
        cos_angle = cos(self.angle_of_incidence)
        if plane == "tangential":
            effective_radius = self.radius_of_curvature * cos_angle
        else:
            effective_radius = self.radius_of_curvature / cos_angle
        return curved_mirror_reflection_matrix(effective_radius)


@dataclass(frozen=True)
class CustomABCD(OpticalElement):
    A: float
    B: float
    C: float
    D: float

    def matrix(self, plane: Plane = "tangential") -> ABCDMatrix:
        _validate_plane(plane)
        return custom_matrix(self.A, self.B, self.C, self.D)


def _validate_plane(plane: str) -> None:
    if plane not in PLANES:
        raise ElementError(f"Unknown propagation plane '{plane}'.")


def _validate_power_coefficients(name: str, reflectivity: float, transmission: float, loss: float) -> None:
    values = {
        "reflectivity": reflectivity,
        "transmission": transmission,
        "loss": loss,
    }
    for label, value in values.items():
        if value < -ABS_TOL:
            raise ElementError(f"Mirror {name} has negative {label}.")
        if value > 1.0 + ABS_TOL:
            raise ElementError(f"Mirror {name} has {label} greater than 1.")

    total = reflectivity + transmission + loss
    if abs(total - 1.0) > max(ABS_TOL, REL_TOL * max(abs(total), 1.0)):
        raise ElementError(f"Mirror {name} has R + T + L != 1.")
    if reflectivity < -ABS_TOL:
        raise ElementError(f"Mirror {name} has T + L > 1, giving negative reflectivity.")
