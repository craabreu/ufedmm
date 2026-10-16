# Development, testing, and deployment tools

This directory holds the conda environments, the conda recipe and the helper scripts
used by the GitHub Actions workflows in `.github/workflows`.

## Manifest

### Conda environments (`conda-envs`)

* `test_env.yaml`: runtime dependencies plus pytest and pytest-cov. Used by the Linux,
  MacOS and Windows test workflows.
* `lint_env.yaml`: runtime dependencies plus ruff. Used by the Linter workflow.
* `docs_env.yaml`: runtime dependencies, pytest and the Sphinx packages. Used by the Docs
  workflow.
* `deployment_env.yaml`: runtime dependencies plus conda-build, anaconda-client, build
  and twine. Used by the Anaconda and PyPI workflows.

The CI workflows pin the Python and OpenMM versions when they create these environments.

### Conda recipe (`conda-recipes/anaconda`)

`meta.yaml` builds the noarch conda package uploaded to the
[mdtools](https://anaconda.org/mdtools) channel. It reads the version from the
`UFEDMM_VERSION` environment variable, which must hold the release tag (for example,
`v0.4.0`).

### Scripts (`scripts`)

* `format_and_check.sh`: formats the package with `ruff format` and fixes what
  `ruff check --fix` can fix.
* `create_conda_env.py`: creates a conda environment from one of the files in
  `conda-envs`, with command-line options for the Python version and environment name.

## Versioning

The version comes from git tags through
[versioningit](https://versioningit.readthedocs.io). Building or installing the package
writes it to `ufedmm/_version.py`, which is also committed. Before each release, set
`__version__` in `ufedmm/_version.py` to the new version. The Anaconda and PyPI
workflows refuse to deploy when that file does not match the release tag.

## Releasing

1. Bump `ufedmm/_version.py`, `CHANGELOG.md` and `CITATION.cff`, and merge to `main`.
2. Publish a GitHub Release with a `vX.Y.Z` tag. This triggers the Docs, Anaconda and
   PyPI workflows. Prereleases are not deployed.
3. To rebuild an existing release, run the corresponding workflow manually and give it
   the tag.
