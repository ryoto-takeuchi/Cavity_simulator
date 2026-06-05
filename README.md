# cavity-designer

`cavity-designer` is a lightweight optical cavity design tool based on ABCD
matrices and Gaussian beam propagation. It reads a YAML cavity description,
computes stability and eigenmodes, reports basic cavity metrics, and can save
beam propagation and 2D layout plots.

This project is intended for quick design checks in an optics or atomic-physics
lab. It is not a full FFT propagation package and does not model coating thermal
noise, PDH signals, mirror surface maps, polarization, birefringence, or
higher-order mode decomposition.

## Installation

Create and activate a virtual environment, then install the package in editable
mode with the test dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[dev]"
```

From another shell, the same idea is:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## CLI Usage

Run a cavity file and print a summary:

```powershell
.\.venv\Scripts\python -m cavity_designer.cli examples\two_mirror.yaml --summary
```

Save beam-radius, wavefront-curvature, g1-g2 where applicable, and layout plots:

```powershell
.\.venv\Scripts\python -m cavity_designer.cli examples\two_mirror.yaml --summary --plot --output outputs\two_mirror
```

After installation, the console entry point is also available:

```powershell
cavity-design examples\two_mirror.yaml --summary --plot --output outputs\two_mirror
```

## Example Config

```yaml
wavelength: 698 nm

layout:
  start: [0 mm, 0 mm]
  direction: 0 deg
  mirror_size: 12.7 mm
  beam_radius_scale: 25

cavity:
  type: linear

elements:
  - name: M1
    type: mirror
    R: inf
    AOI: 0 deg
    T: 100 ppm
    L: 10 ppm

  - name: space_1
    type: space
    length: 100 mm

  - name: M2
    type: curved_mirror
    Rc: 200 mm
    AOI: 0 deg
    T: 10 ppm
    L: 10 ppm
```

`examples/` includes:

- `two_mirror.yaml`: linear plano-concave reference cavity
- `plano_concave.yaml`: another two-mirror linear example
- `intracavity_lens.yaml`: linear cavity with a thin intracavity lens
- `ring_cavity.yaml`: three-mirror ring cavity
- `ring_cavity_paper.yaml`: three-mirror ring reproducing the 198.002 mm paper geometry
- `bow_tie.yaml`: four-mirror bow-tie ring cavity
- `bow-tie_w_expander.yaml`: folded linear path with a mirror expander

All example files include a `layout` section so `layout.png` can be generated.

## Units

Length-like quantities require explicit units. Supported length units are:

```text
nm, um, mm, cm, m
```

Bare numbers are accepted only for dimensionless quantities. Mirror power
quantities accept fractions, `ppm`, or `%`.

User-facing `R`, `T`, and `L` mean power reflectivity, power transmission, and
power loss. If `R` is omitted, the parser computes:

```text
R = 1 - T - L
```

and raises an error if the result is negative.

Angles require explicit units:

```text
deg, mrad, rad
```

If `AOI` is omitted for a mirror, it defaults to `0 deg`.

## Supported Elements

- `space`: free-space propagation, `[[1, L], [0, 1]]`
- `thin_lens`: thin lens, `[[1, 0], [-1/f, 1]]`
- `mirror` or `flat_mirror`: flat mirror reflection, identity ABCD matrix
- `curved_mirror`: curved mirror reflection
- `custom_abcd`: explicit `A`, `B`, `C`, `D`

Curved mirror reflection at normal incidence uses:

```text
[[1, 0],
 [-2/Rc, 1]]
```

`Rc` is signed. Positive `Rc` focuses on reflection, and negative `Rc`
defocuses on reflection, as for a convex mirror.

Thin lenses support scalar and plane-specific focal lengths:

```yaml
- name: thermal_lens
  type: thin_lens
  f: 100 mm
  L: 100 ppm

- name: cylindrical_lens
  type: thin_lens
  f_tangential: 100 mm
  f_sagittal: 150 mm
  L: 50 ppm
```

Lens `L` is a per-pass power loss. A lens in a linear cavity is counted twice
per round trip; a lens in a ring cavity is counted once per round trip.

## Linear And Ring Paths

The reference plane is immediately after the first element. For a linear cavity,
the round-trip path is formed by going forward through the element list and then
returning through the same elements in reverse order. For example:

```text
M1 -> space -> M2 -> space -> M1
```

For a ring cavity, the path follows the element list once and returns to the
first element:

```text
M1 -> arm_12 -> M2 -> arm_23 -> M3 -> arm_31 -> M1
```

The CLI prints both the physical path and the exact ABCD sequence used for the
round-trip calculation.

## Sagittal And Tangential Planes

The tool computes sagittal and tangential results separately. For oblique
spherical mirrors, it uses the effective radii:

```text
R_tangential = Rc cos(theta)
R_sagittal   = Rc / cos(theta)
```

The AOI sign affects only the 2D layout branch. The ABCD matrices use
`cos(theta)`, so changing the AOI sign does not change the sagittal or
tangential focusing strength.

## Layout Plotting

The optional `layout` section controls the 2D drawing:

```yaml
layout:
  start: [0 mm, 0 mm]
  direction: 0 deg
  mirror_size: 12.7 mm
  beam_radius_scale: 25
```

`start` and `direction` choose the initial ray position and direction.
`mirror_size` sets a common drawn aperture length for all mirrors.
`beam_radius_scale` multiplies the plotted Gaussian beam-radius envelope; use
`1` for true scale, and larger values when micrometer-scale beams should be
visible on centimeter-scale layouts.

For mirror placement, the signed AOI selects one of the two possible 2D
reflection branches:

```text
mirror_tangent_angle = theta_in + pi/2 - AOI
theta_out = theta_in + pi - 2 AOI
```

If the same mirror is encountered again in a folded linear path, the previously
inferred mirror tangent is reused. This lets paths such as:

```text
M1 -> M2 -> M3 -> M4 -> M3 -> M2 -> M1
```

draw as a retraced physical layout.

Curved mirrors are drawn as circular arcs using the signed `Rc` and
`layout.mirror_size` as the chord length. The layout beam envelope currently
uses the tangential eigenmode.

## Gaussian Convention

The complex beam parameter convention is:

```text
1/q = 1/R - i lambda/(pi w^2)
```

With this convention, a freely propagating waist has:

```text
q = z + i z_R
```

and the physical eigenmode has:

```text
Im(q) > 0
```

Beam radius is computed from:

```text
w = sqrt(-lambda / (pi imag(1/q)))
```

because `imag(1/q)` is negative for a physical Gaussian beam.

## Stability

For a general round-trip ABCD matrix:

```text
M = [[A, B],
     [C, D]]
```

the stability condition is:

```text
abs((A + D) / 2) < 1
```

This trace condition is used for all cavity types, including three-mirror and
four-mirror cavities. For two-mirror linear cavities, the CLI also reports the
familiar diagnostic:

```text
g1 = 1 - L/R1
g2 = 1 - L/R2
stable if 0 < g1*g2 < 1
```

## Metrics And Build-Up

The CLI reports round-trip length, FSR, round-trip power loss, finesse,
linewidth, cavity pole, photon lifetime, round-trip Gouy phase shift, and
build-up estimates.

FSR is computed as:

```text
linear cavity: FSR = c / (2 L)
ring cavity:   FSR = c / L_rt
```

The high-finesse finesse approximation is:

```text
F ~ 2*pi / loss_rt
```

For a simple two-mirror linear cavity, the on-resonance build-up estimate uses
the coherent high-finesse expression:

```text
B ~ 4*T1 / (T1 + T2 + Lrt)^2
```

For ring cavities, the current MVP intentionally uses a one-direction
traveling-wave power sum rather than a coherent standing-wave interference
model:

```text
eta_rt = product(R_i)
B_ring = T_input / (1 - eta_rt)
```

where `R_i = 1 - T_i - L_i` for each mirror.

## Tests

Run the test suite with:

```powershell
.\.venv\Scripts\python -m pytest
```

The tests cover unit parsing, matrix conventions, Gaussian eigenmodes,
two-mirror analytic checks, ring cavities, intracavity lenses, layout geometry,
metrics, and CLI behavior.

## Current Limitations

- No FFT propagation
- No arbitrary mirror surface maps
- No coating thermal-noise model
- No PDH error-signal simulation
- No polarization, birefringence, or mode-decomposition model
- Layout plots are geometric sketches and do not enforce mechanical clearance
- Layout beam envelopes currently use the tangential eigenmode only
