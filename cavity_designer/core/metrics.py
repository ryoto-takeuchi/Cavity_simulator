"""Cavity performance metrics."""

from __future__ import annotations

from dataclasses import dataclass

from cavity_designer.core.constants import ABS_TOL, SPEED_OF_LIGHT
from cavity_designer.core.cavity import Cavity
from cavity_designer.core.elements import Mirror


@dataclass(frozen=True)
class CavityMetrics:
    optical_path_length: float
    geometric_round_trip_length: float
    fsr: float
    round_trip_power_loss: float
    finesse: float | None
    linewidth: float | None
    cavity_pole: float | None
    photon_lifetime: float | None
    power_buildup: float | None
    coupling_regime: str
    warnings: tuple[str, ...] = ()


def compute_metrics(cavity: Cavity) -> CavityMetrics:
    round_trip_length = cavity.round_trip_length
    if round_trip_length <= 0:
        raise ValueError("Round-trip length must be positive.")
    fsr = SPEED_OF_LIGHT / round_trip_length
    loss_rt = cavity.round_trip_power_loss()

    finesse = None
    linewidth = None
    cavity_pole = None
    photon_lifetime = None
    if loss_rt > ABS_TOL:
        finesse = 2.0 * 3.141592653589793 / loss_rt
        linewidth = fsr / finesse
        cavity_pole = linewidth / 2.0
        photon_lifetime = 1.0 / (2.0 * 3.141592653589793 * linewidth)

    buildup = None
    coupling = "not estimated"
    warnings: list[str] = []
    if cavity.cavity_type == "linear" and len(cavity.mirrors) == 2:
        mirrors = cavity.mirrors
        input_transmission = mirrors[0].power_transmission
        end_transmission = mirrors[1].power_transmission
        internal_loss = max(loss_rt - input_transmission - end_transmission, 0.0)
        total_coupling_loss = input_transmission + end_transmission + internal_loss
        if total_coupling_loss > ABS_TOL:
            buildup = 4.0 * input_transmission / total_coupling_loss**2
        other_loss = end_transmission + internal_loss
        if abs(input_transmission - other_loss) <= max(10 * ABS_TOL, 0.05 * max(input_transmission, other_loss, ABS_TOL)):
            coupling = "near critical"
        elif input_transmission < other_loss:
            coupling = "under-coupled"
        else:
            coupling = "over-coupled"
    elif cavity.cavity_type == "ring":
        input_mirror = _ring_input_mirror(cavity)
        if input_mirror is None:
            warnings.append("Ring power build-up estimate requires at least one mirror.")
        else:
            survival = _ring_round_trip_power_survival(cavity)
            denominator = 1.0 - survival
            if denominator > ABS_TOL:
                buildup = input_mirror.power_transmission / denominator
                coupling = "one-way traveling-wave power sum"
            else:
                warnings.append("Ring round-trip power loss is too small to estimate build-up robustly.")
            if input_mirror is not cavity.elements[0]:
                warnings.append("Ring build-up uses the first mirror in the element list as the input coupler when possible.")
    else:
        warnings.append("Power build-up estimate is only implemented for two-mirror linear cavities and ring cavities.")

    return CavityMetrics(
        optical_path_length=cavity.optical_path_length(),
        geometric_round_trip_length=round_trip_length,
        fsr=fsr,
        round_trip_power_loss=loss_rt,
        finesse=finesse,
        linewidth=linewidth,
        cavity_pole=cavity_pole,
        photon_lifetime=photon_lifetime,
        power_buildup=buildup,
        coupling_regime=coupling,
        warnings=tuple(warnings),
    )


def _ring_input_mirror(cavity: Cavity) -> Mirror | None:
    if cavity.elements and isinstance(cavity.elements[0], Mirror):
        return cavity.elements[0]
    for element in cavity.elements:
        if isinstance(element, Mirror):
            return element
    return None


def _ring_round_trip_power_survival(cavity: Cavity) -> float:
    survival = 1.0
    for element in cavity.sequence_for_round_trip():
        if isinstance(element, Mirror):
            survival *= element.power_reflectivity
        else:
            survival *= max(0.0, 1.0 - element.power_transmission - element.power_loss)
    return survival
