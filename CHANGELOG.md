# Changelog

## 0.4.0

### Breaking changes
- UFEDMM now requires Python 3.10 or newer and OpenMM 8.4 or newer.

### Added
- UFEDMM is published on PyPI (`pip install ufedmm`) as well as on the `mdtools`
  conda channel.
- `ufedmm.__version__` reports the installed version.
- `ExtendedSpaceState.getVelocities` takes a `split` argument. With `extended=True,
  split=True`, it returns the velocities of the physical particles and the x
  velocities of the extra particles separately.
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
- `MiddleMassiveNHCIntegrator`, `MiddleMassiveGGMTIntegrator`,
  `RegulatedNHLIntegrator`, and integrators created with `num_rattles=0` now update
  virtual sites after moving particles in the middle of a step. On the CUDA and
  OpenCL platforms, forces were computed with outdated virtual-site positions, which
  made systems with TIP4P-type water heat up uncontrollably (#37).
- UFEDMM simulations raise a `ValueError` when the system has constraints and the
  integrator has no RATTLE steps (`MiddleMassiveNHCIntegrator`,
  `MiddleMassiveGGMTIntegrator`, `RegulatedNHLIntegrator`, or `num_rattles=0`).
  Those integrators used to ignore the constraints silently.
- `StateDataReporter` with `multipleTemperatures=True` works again on OpenMM 8.3 and
  newer. It used to fail because the reported state had no velocities or energies.
- `FreeEnergyAnalyzer.mean_force_free_energy` works again with NumPy 2 and
  OpenMM 8, and its fit now uses each collective variable's own kernel width and
  period. Before this fix, every kernel used the last variable's values.
