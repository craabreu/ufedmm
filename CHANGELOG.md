# Changelog

## 0.4.0

### Breaking changes
- UFEDMM now requires Python 3.10 or newer and OpenMM 8.4 or newer.

### Added
- UFEDMM is published on PyPI (`pip install ufedmm`) as well as on the `mdtools`
  conda channel.
- `ufedmm.__version__` reports the installed version.
- A `CITATION.cff` file points to the UFEDMM paper (Bajpai et al., J. Comput.
  Chem. 2023).

### Changed
- The documentation moved to https://craabreu.github.io/ufedmm/latest, with a
  separate copy for each release. Old deep links into the documentation no
  longer resolve.
- Releases are deployed when a GitHub Release is published.
- The code is formatted and linted with ruff.

### Fixed
- `FreeEnergyAnalyzer.mean_force_free_energy` works again with NumPy 2 and
  OpenMM 8, and its fit now uses each collective variable's own kernel width and
  period. Before this fix, every kernel used the last variable's values.
