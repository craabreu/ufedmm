# Changelog

## 0.4.0

### Breaking changes
- UFEDMM now requires Python 3.10 or newer and OpenMM 8.4 or newer.
- `MiddleMassiveNHCIntegrator` no longer takes a `track_energy` argument. It only
  added per-dof variables that were never updated, so it had no effect.

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
- Gridded metadynamics with two or three bias variables deposits hills of the
  requested height. Each hill used to have height `h**n` for `n` bias variables
  (e.g., `h**2` in kJ/mol for a 2D bias), so results obtained with a
  multidimensional bias should be recomputed. One-dimensional and gridless
  metadynamics were not affected.
- Tasks derived from `PeriodicTask` that do not override `initialize` and
  `loadCheckpoint` can be added with `ExtendedSpaceSimulation.add_periodic_task`.
  They used to fail with a `TypeError`, because those base-class methods took
  different arguments from the ones the simulation passes.
- Loading a gridless metadynamics checkpoint into a simulation that has deposited
  more hills than the checkpoint holds keeps the restored hills. It used to erase
  the first hills instead of the extra ones.
- `ExtendedSpaceSimulation.saveCheckpoint` and `loadCheckpoint` work with any
  binary file object, such as `io.BytesIO`. With metadynamics, they used to require
  a file on disk. The checkpoint format is unchanged.
- `FreeEnergyAnalyzer.metadynamics_bias_free_energy` returns the right function
  when called more than once. From the second call on, it used to ignore all
  variables and return minus the sum of the hill heights everywhere.
- Well-tempered metadynamics works without a grid, that is, with
  `enforce_gridless=True` or with more than three bias variables. It used to fail
  when depositing the first hill.
- Gridded metadynamics that mixes periodic and non-periodic bias variables
  deposits hills in the right place. The periodic padding of each hill was applied
  along the wrong array axis, which scrambled the hills, so results obtained with
  such a bias should be recomputed.
- Extra particles no longer change the Born radii of a `CustomGBForce` (such as the
  GBn2 implicit solvent), so they no longer change the solvation energy of the
  physical system.
- `FreeEnergyAnalyzer.centers_and_mean_forces` works with a DataFrame whose index is
  not `0, 1, ..., n-1`, such as a slice of a longer trajectory. It used to raise an
  `IndexError`, or with a shuffled index, pair forces with the wrong samples.
- `ExtendedSpaceContext.setPositions` accepts positions as a NumPy array, such as
  `State.getPositions(asNumpy=True)`.
- `Tee` closes only the files it opened from file names, not the streams passed to
  it.
