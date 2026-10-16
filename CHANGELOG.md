# Changelog

## 0.4.0

### Breaking changes
- UFEDMM now requires Python 3.10 or newer and OpenMM 8.4 or newer.
- `ExtendedSpaceState.getVelocities(extended=True)` now returns a tuple with the
  velocities of the physical particles and those of the dynamical variables, like
  `getPositions(extended=True)`. Pass `split=False` to get a single list as before.

### Added
- UFEDMM is published on PyPI (`pip install ufedmm`) as well as on the `mdtools`
  conda channel.
- `ufedmm.__version__` reports the installed version.
- A `CITATION.cff` file points to the UFEDMM paper (Bajpai et al., J. Comput.
  Chem. 2023).

### Changed
- `ExtendedSpaceContext.setVelocitiesToTemperature` removes the center-of-mass
  velocity of the physical particles, applies the constraints, and scales the
  velocities to the exact target temperature. Massless particles get zero velocity.
  The extra particles only get velocities along x, drawn at each variable's own
  temperature.
- The UFEDMM integrators no longer thermostat the y and z directions of the extra
  particles, which only move along x.
- The documentation moved to https://craabreu.github.io/ufedmm/latest, with a
  separate copy for each release. Old deep links into the documentation no
  longer resolve.
- Releases are deployed when a GitHub Release is published.
- The code is formatted and linted with ruff.
- Removed code paths for OpenMM versions older than 8.4. They compared version
  strings, so a future OpenMM 10 would have been treated as older than 7.5.
- The examples import `openmm` directly, not the deprecated `simtk.openmm`.

### Fixed
- `StateDataReporter` with `multipleTemperatures=True` works again on OpenMM 8.3 and
  newer. It used to fail because the reported state had no velocities or energies.
- `FreeEnergyAnalyzer.mean_force_free_energy` works again with NumPy 2 and
  OpenMM 8, and its fit now uses each collective variable's own kernel width and
  period. Before this fix, every kernel used the last variable's values.
