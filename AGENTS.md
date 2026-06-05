# AGENTS.md

## Project Overview

This repository implements a lightweight optical cavity design tool.

The goal is to let a researcher describe an optical resonator in a text-based configuration file and obtain:

* cavity stability
* round-trip ABCD matrix
* Gaussian eigenmode
* beam radius and wavefront curvature along the optical path
* FSR, finesse, linewidth, enhancement, photon lifetime
* basic impedance-matching diagnostics
* stability plots and beam propagation plots

The first target is a robust MVP for Gaussian-beam / ABCD-matrix cavity design. Do not implement FFT propagation, coating thermal noise, PDH signal simulation, or full GUI features unless explicitly requested later.

## Main Design Principle

Prioritize physical correctness, clear conventions, and testability over feature count.

The tool should be useful for checking real optical-cavity designs in an atomic-physics laboratory. Avoid hidden sign conventions, ambiguous reflectivity definitions, or unit assumptions.

## Expected Tech Stack

Use Python.

Preferred dependencies:

* `numpy`
* `scipy`
* `matplotlib`
* `pydantic` or dataclasses for structured models
* `pyyaml` or `tomli` / `tomllib` for config parsing
* `pytest` for tests

Optional but useful:

* `pint` for units, if it does not overcomplicate the MVP

Avoid heavy GUI frameworks in the MVP.

## Repository Structure

Use a structure close to:

```text
cavity_designer/
  __init__.py
  core/
    __init__.py
    matrix.py
    elements.py
    gaussian.py
    cavity.py
    metrics.py
  io/
    __init__.py
    parser.py
  plot/
    __init__.py
    beam_plot.py
    stability_plot.py
  cli.py

examples/
  two_mirror.yaml
  plano_concave.yaml
  ring_cavity.yaml

tests/
  test_matrix.py
  test_gaussian.py
  test_two_mirror_cavity.py
  test_metrics.py

README.md
AGENTS.md
pyproject.toml
```

## MVP Features

Implement the following first.

### 1. Text-Based Cavity Description

Support a YAML configuration format.

Example:

```yaml
wavelength: 698 nm

cavity:
  type: linear

elements:
  - name: M1
    type: mirror
    R: inf
    T: 100 ppm
    L: 10 ppm

  - name: space_1
    type: space
    length: 100 mm

  - name: M2
    type: curved_mirror
    Rc: 200 mm
    T: 10 ppm
    L: 10 ppm
```

Use explicit units. At minimum support:

* length: `nm`, `um`, `mm`, `cm`, `m`
* dimensionless fractional quantities: raw float
* optical loss/transmission: `ppm`, `%`, raw fraction

Internally convert all lengths to meters and all power coefficients to dimensionless fractions.

### 2. Optical Elements

Implement at least:

* `Space(length)`
* `ThinLens(f)`
* `FlatMirror(T, L)`
* `CurvedMirror(Rc, T, L)`
* `CustomABCD(A, B, C, D)`

Use ray-transfer matrices in one transverse dimension.

For the MVP, use the paraxial ABCD formalism.

Typical matrices:

```text
Space(L):
[[1, L],
 [0, 1]]

ThinLens(f):
[[1, 0],
 [-1/f, 1]]

Curved mirror reflection:
[[1, 0],
 [-2/Rc, 1]]
```

Be explicit in the code and documentation that the mirror matrix is for reflection.

### 3. Round-Trip Matrix

Compute the round-trip ABCD matrix for the cavity.

For a general round-trip matrix

```text
M = [[A, B],
     [C, D]]
```

the stability condition is:

```text
abs((A + D) / 2) < 1
```

For two-mirror cavities, also provide the familiar diagnostic:

```text
g1 = 1 - L / R1
g2 = 1 - L / R2
stable if 0 < g1 * g2 < 1
```

But do not use only the `g1 g2` condition for general cavities.

### 4. Gaussian Eigenmode

Use the complex beam parameter convention:

```text
1/q = 1/R - i * lambda / (pi * w^2)
```

With this convention, a freely propagating waist has:

```text
q = z + i z_R
```

The physical solution should have:

```text
Im(q) > 0
```

For a round-trip matrix, solve:

```text
q = (A q + B) / (C q + D)
```

equivalently:

```text
C q^2 + (D - A) q - B = 0
```

Return no physical eigenmode if:

* the cavity is unstable
* the quadratic equation has no solution with `Im(q) > 0`
* numerical precision makes the result invalid

### 5. Beam Propagation

Given the cavity eigenmode at a reference plane, propagate q through each element and sample along spaces.

For each sampled position, compute:

```text
w(z)
R(z)
Gouy phase
q(z)
```

Using the selected convention:

```text
1/q = 1/R - i * lambda / (pi * w^2)
```

The beam radius should be computed from:

```text
w = sqrt(-lambda / (pi * imag(1/q)))
```

because `imag(1/q)` is negative for a physical Gaussian beam in this convention.

Handle infinite curvature at waist cleanly.

### 6. Cavity Metrics

Compute and report:

* optical path length
* geometric round-trip length
* FSR
* finesse
* linewidth
* cavity pole / photon lifetime estimate
* power build-up / enhancement
* impedance matching status

For a simple linear cavity:

```text
FSR = c / (2 L)
```

For a ring cavity:

```text
FSR = c / L_rt
```

where `L_rt` is the round-trip path length.

For high-finesse approximations, document assumptions clearly.

A reasonable first implementation for round-trip power loss is:

```text
loss_rt = sum(transmissions and losses per round trip)
finesse ≈ 2*pi / loss_rt
```

For a two-mirror cavity with input mirror transmission `T1`, end mirror transmission `T2`, and additional round-trip loss `Lrt`, estimate power build-up on resonance as:

```text
B ≈ 4*T1 / (T1 + T2 + Lrt)^2
```

Only use this approximation when its assumptions are satisfied. Warn when the cavity is not a simple two-mirror cavity.

### 7. Plots

Implement plotting functions:

1. Beam radius along optical path
2. Wavefront curvature along optical path
3. Stability map for two scanned parameters
4. g1-g2 plane for two-mirror cavities

Beam-radius plots should mark optical-element positions with vertical lines and labels.

Use SI units internally, but plots may display convenient units such as mm, cm, or um.

### 8. CLI

Provide a CLI entry point.

Example usage:

```bash
cavity-design examples/two_mirror.yaml --summary --plot
```

Expected behavior:

* parse the config
* validate physical inputs
* compute cavity eigenmode
* print a readable summary
* save plots to an output directory

Example:

```bash
cavity-design examples/two_mirror.yaml --output outputs/two_mirror
```

## Physics and Convention Requirements

### Reflectivity, Transmission, and Loss

User-facing inputs `R`, `T`, and `L` must be power quantities, not field-amplitude quantities.

For power quantities:

```text
R + T + L = 1
```

When field amplitudes are needed internally:

```text
r = sqrt(R)
t = sqrt(T)
```

Never silently mix power reflectivity and amplitude reflectivity.

If a mirror specifies `T` and `L` but not `R`, compute:

```text
R = 1 - T - L
```

Raise an error if this is negative.

### Units

Never assume that a bare number in a config is millimeters or nanometers.

Allowed behavior:

* bare number for dimensionless fractions
* explicit unit required for length-like values
* explicit support for `ppm`, `%`, and fractions for losses/transmissions

### Sign Conventions

Document all sign conventions in `README.md`.

Use the q-parameter convention consistently:

```text
1/q = 1/R - i lambda/(pi w^2)
```

Use `Im(q) > 0` as the physical solution criterion.

### Numerical Robustness

Use tolerances where needed.

Recommended defaults:

```python
ABS_TOL = 1e-12
REL_TOL = 1e-10
```

Avoid equality comparisons for floats.

## Testing Requirements

Every physics formula must have at least one unit test.

Minimum tests:

### Matrix Tests

* Space matrix
* Thin lens matrix
* Curved mirror reflection matrix
* Matrix multiplication order

### Two-Mirror Cavity Tests

Test against analytic two-mirror cavity formulas.

For a symmetric cavity with length `L` and mirror radius `R`, verify:

```text
0 < g^2 < 1
```

and compare computed waist size to the known analytic expression.

### Stability Tests

Test:

* stable cavity returns a physical q
* unstable cavity returns no physical q or raises a controlled exception
* marginally stable cavity is handled carefully

### Metrics Tests

Check:

* FSR for linear cavity
* FSR for ring cavity
* finesse approximation for small round-trip loss
* linewidth = FSR / finesse

### Parser Tests

Check:

* `100 ppm`
* `1 %`
* `698 nm`
* `100 mm`
* invalid unit
* missing required field
* negative loss
* `T + L > 1`

## Coding Style

Use clear, explicit code.

Prefer:

* dataclasses or pydantic models for optical elements
* type hints
* small functions
* deterministic behavior
* explicit errors

Avoid:

* hidden global state
* silent unit conversions
* ambiguous names like `R` when it could mean reflectivity or radius of curvature

Use names such as:

```python
radius_of_curvature
power_reflectivity
power_transmission
power_loss
```

rather than ambiguous short names in internal APIs.

Short names like `R`, `T`, `L`, and `Rc` may be accepted in YAML input but should be normalized internally.

## Documentation Requirements

Create or update `README.md` with:

* project purpose
* installation
* example config
* CLI usage
* sign conventions
* supported elements
* limitations of the MVP
* explanation of stability condition
* explanation of q-parameter convention

The README should state that this is an ABCD/Gaussian-mode design tool, not a full FFT optical simulation package.

## Error Handling

Provide clear error messages.

Examples:

```text
Mirror M1 has T + L > 1, giving negative reflectivity.
Length value '100' requires an explicit unit.
No stable Gaussian eigenmode exists for this cavity.
The g1-g2 plot is only available for two-mirror cavities.
```

Do not fail with obscure linear-algebra or parsing tracebacks when the input is physically invalid.

## Scope Boundaries for MVP

Do not implement in the MVP unless explicitly requested:

* FFT propagation
* arbitrary mirror surface maps
* coating thermal noise
* vibration sensitivity
* PDH error-signal simulation
* higher-order Hermite-Gaussian mode decomposition
* polarization effects
* birefringence
* astigmatism from oblique incidence
* GUI
* web app
* database storage

However, design the internal API so that sagittal/tangential ABCD propagation can be added later.

## Future Extension Hooks

Keep the architecture extensible for:

* ring cavities
* bow-tie cavities
* sagittal/tangential propagation
* astigmatism from oblique-incidence mirrors
* mode matching from an input beam
* parameter scans and optimization
* aperture and clipping-loss estimates
* PDH-related diagnostics
* thermal lens elements

## Development Order

Implement in this order:

1. Unit parser
2. ABCD matrix utilities
3. Optical element classes
4. Round-trip matrix calculation
5. Stability check
6. Gaussian eigenmode solver
7. Beam propagation sampler
8. Cavity metrics
9. CLI summary output
10. Beam-radius plot
11. Curvature plot
12. g1-g2 plot for two-mirror cavities
13. Tests
14. README examples

Do not start with GUI work.

## Acceptance Criteria for First Working Version

The MVP is complete when:

* `examples/two_mirror.yaml` runs from the CLI
* the program prints cavity stability, q parameter, waist size, FSR, finesse, linewidth, and enhancement
* beam radius is plotted along the cavity path
* mirror positions are visible in the plot
* unstable cavity examples fail gracefully
* tests pass with `pytest`
* all formulas used in the MVP are documented in `README.md`

## Example CLI Output Target

The CLI summary should look approximately like:

```text
Cavity summary
--------------
Type: linear
Wavelength: 698.000 nm
Round-trip length: 0.200000 m
FSR: 1.499 GHz

Stability
---------
Round-trip ABCD:
A = ...
B = ...
C = ...
D = ...
Trace condition: |(A+D)/2| = ...
Stable: yes

Gaussian mode
-------------
q at reference plane: ... + ...j m
waist radius: ... um
waist position: ... mm from reference
Rayleigh range: ... mm

Cavity metrics
--------------
Round-trip power loss: ... ppm
Finesse: ...
Linewidth: ... kHz
Estimated build-up: ...
Coupling regime: under-coupled / over-coupled / near critical
```

## Important Notes for the Coding Agent

When making implementation choices, prefer the simplest physically correct implementation.

If a requested feature conflicts with the sign convention, unit convention, or power-vs-amplitude convention above, stop and adjust the implementation to preserve these conventions.

Never silently reinterpret user input.

Always add or update tests when changing physics logic.
