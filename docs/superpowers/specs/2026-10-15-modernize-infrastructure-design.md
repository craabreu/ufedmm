# Modernize ufedmm infrastructure after openxps

Date: 2026-10-15

## Goal

Bring ufedmm's tooling (packaging, CI, linting, release deployment and docs) up to
the current state of [openxps](https://github.com/craabreu/openxps). Both projects
started from the same cookiecutter template, so each ufedmm file is replaced by, or
aligned with, its openxps counterpart. The science code doesn't change in
behavior; the only source edits are a mechanical ruff restyle (PR 2).

## Decisions

- **Supported versions:** Python >= 3.10 and OpenMM >= 8.4, the same as openxps.
  The CI test matrix is copied from openxps: Linux 3.14/8.6, 3.13/8.6, 3.12/8.5,
  3.11/8.4 and 3.10/8.4, plus one macOS row and one Windows row at 3.13/8.6.
- **Version file:** `ufedmm/_version.py` is committed and bumped by hand before
  each release, as in openxps. `ufedmm.__version__` is exported.
- **Releases:** deployment runs when a GitHub Release is published, not when a tag
  is pushed. Prereleases are skipped. `workflow_dispatch` takes a tag, for
  rebuilding a past release.
- **Package channels:** conda packages go to the `mdtools` channel. ufedmm is
  published on PyPI for the first time; the name is currently free.
- **cvpack:** the canonical location is https://github.com/craabreu/cvpack, and its
  docs are at https://craabreu.github.io/cvpack/latest. cvpack is on PyPI, so it
  is declared as a real dependency.
- **Citation:** the preferred citation is S. Bajpai, B. K. Petkov, M. Tong,
  C. R. A. Abreu, N. N. Nair and M. E. Tuckerman, *J. Comput. Chem.* **44**(28),
  2166–2183 (2023), doi:10.1002/jcc.27182.
- **Workspace:** the work is done in the shared checkout
  `/data3/craabreu/github_repos/ufedmm`, which the `openxps-test` env imports as
  an editable install.
- **Delivery:** three PRs, merged in order.

## PR 1: CI, packaging and lint config (branch `modernize-ci-packaging`)

### `pyproject.toml`
- Set `requires-python = ">=3.10"` and add `license = { text = "MIT" }`.
- Classifiers: `License :: OSI Approved :: MIT License` and Python 3.10–3.14.
  Remove `Other/Proprietary License`.
- Dependencies: `cvpack`, `numpy`, `openmm>=8.4`, `pandas`, `pyyaml` and `scipy`.
  Remove the commented-out entries.
- Test extras: `pytest>=7.0` and `pytest-cov`. Remove `pytest-runner`.
- URLs: Source, Documentation (`https://craabreu.github.io/ufedmm/latest`) and
  Changelog (`.../blob/main/CHANGELOG.md`).
- Replace the `[tool.isort]` and `[tool.flake8]` sections with openxps's
  `[tool.ruff]`, `[tool.ruff.lint]`, `[tool.ruff.lint.pylint]`,
  `[tool.ruff.lint.per-file-ignores]` and `[tool.ruff.format]` sections. Keep
  `extend-exclude = ["_version.py"]`.
- Add `--cov-report=xml` to the pytest `addopts`.
- Package data: keep the existing entry, and check that it matches the files that
  are actually in `ufedmm/data/`.

### Version
- Keep `ufedmm/_version.py` under version control, holding `__version__ = "0.3.2"`.
- Add `from ._version import __version__  # noqa: F401` to `ufedmm/__init__.py`.
- Add a `setup.cfg` that copies openxps's `[coverage:run]` omit list (tests and
  `_version.py`) and its `[aliases]` section. Drop the yapf and flake8 sections.

### Workflows
- Linux, MacOS and Windows: openxps's files, with `ufedmm` substituted. They use
  `actions/checkout@v4`, `mamba-org/setup-micromamba@v2` with `create-args`,
  `fail-fast: false`, plain `pytest` (its options come from `addopts`) and
  `codecov/codecov-action@v4` with `files: ./coverage.xml`.
- Linter: openxps's file, at Python 3.12, running `ruff check ufedmm`.

### Conda environments (`devtools/conda-envs/`)
- `test_env.yaml`: base dependencies plus pytest and pytest-cov. Remove codecov
  and the sphinx packages.
- `lint_env.yaml`: base dependencies plus `ruff`. Remove flake8,
  flake8-pyproject, black and isort.
- `docs_env.yaml`: base dependencies, pytest, pytest-cov, sphinx,
  sphinxcontrib-bibtex and sphinx-copybutton. The Docs workflow runs the tests,
  so it needs pytest.
- Replace `conda_build_env.yaml` with `deployment_env.yaml`: base dependencies,
  anaconda-client, conda-build, and the pip packages `build` and `twine`. Drop
  `boa`.

### Conda recipe
- Run requirements: `python >=3.10` and `openmm >=8.4`. Drop the
  `importlib-resources` selector.
- Set `license_file: LICENSE.md` and point `doc_url` at `/latest`.

### Cruft and miscellaneous
- Delete `.readthedocs.yml` and `.pylintrc`. Keep `.codecov.yml`.
- Rename `LICENSE` to `LICENSE.md`.
- `MANIFEST.in`: remove `include versioneer.py`, update the license file name
  and add `include CODE_OF_CONDUCT.md`.
- `devtools/scripts/format_and_check.sh`: switch to `ruff format` and
  `ruff check`, as openxps does. Update `devtools/README.md` to match.
- `.github/CONTRIBUTING.md`: replace any mentions of flake8, black or isort with
  ruff.

### Keeping CI green
PR 1 doesn't run the formatter over the source. Run `ruff check ufedmm` and add
every rule code that fails to a temporary `ignore` list with a
`# TODO(PR2): remove` comment. PR 2 removes them.

### Acceptance criteria
- Locally in `openxps-test`: `pytest` and `ruff check ufedmm` pass, `python -m build`
  produces an sdist and a wheel, and `python -c "import ufedmm; print(ufedmm.__version__)"`
  prints `0.3.2`.
- CI on the PR is green for Linux, MacOS, Windows and the linter.

## PR 2: ruff restyle (branch `ruff-restyle`)

- Run `ruff format` and `ruff check --fix` over `ufedmm/` and `examples/*.py`.
- Fix the remaining violations by hand. Use a targeted `# noqa` only where a fix
  would change behavior or the public API. For example, camelCase method names
  that follow OpenMM conventions stay as they are.
- Remove all `TODO(PR2)` ignores from `pyproject.toml`.
- Add `.git-blame-ignore-revs` holding the SHA of the formatting commit. Keep
  hand-made fixes in a separate commit from the mechanical formatting.

### Acceptance criteria
- The test suite gives the same results before and after (same pass and fail
  counts).
- `ruff format --check ufedmm` and `ruff check ufedmm` both report no problems.
- CI is green.

## PR 3: release-driven deploys, PyPI and docs (branch `release-deploys-docs`)

### Workflows
- `Docs.yaml`: openxps's file, with `UFEDMM_VERSION` in place of
  `OPENXPS_VERSION` and Python 3.13 with OpenMM 8.6. The `REPO_TOKEN` and
  `path: ufedmm` checkout is replaced by `GITHUB_TOKEN`. Pushes to main deploy to
  `docs/development/`. A release, or a `workflow_dispatch` run with a tag,
  deploys to `docs/<tag>/` and `docs/latest/`. Prereleases are skipped.
- `Anaconda.yaml`: openxps's file. It runs on a published release or on
  `workflow_dispatch` with a tag, checks out that tag with `fetch-depth: 0`, runs
  `conda build . -c conda-forge -c mdtools`, and uploads the `.conda` or
  `.tar.bz2` file to `mdtools`.
- `PyPI.yaml`: new, ported from openxps. It runs `python -m build` and then
  `twine upload --skip-existing` with `secrets.PYPI_TOKEN`.

### One-time gh-pages migration
- On the gh-pages branch, delete the flat pages under `docs/` (but keep
  `docs/.nojekyll`). Add `docs/index.html` with a meta refresh to `latest/`,
  identical to the openxps one.
- Seed `latest/` by running the Docs workflow manually on `v0.3.2`. This only
  happens after the user approves.
- Known effect: old deep links such as `/ufedmm/pythonapi/...` will 404. The
  root URL keeps working because it redirects.

### `docs/conf.py`
- The version comes from `os.getenv("UFEDMM_VERSION", ufedmm.__version__)`.
- Port openxps's automatic API-page generator (`create_class_rst_file`,
  `create_function_rst_file` and `create_module_docs`). It writes `docs/api/`,
  which gets a `.gitignore`.
  The modules to document are `ufedmm` (core), `integrators`, `io`, `analysis`
  and `testmodels`. Exclude the `cvpack` re-export.
- Add `extlinks` entries for `OpenMM` and for `CVPack`, which points to
  `https://craabreu.github.io/cvpack/latest/api/%s.html`.
- Rename the leftover template identifiers (`github_repo`, `htmlhelp_basename`,
  and the LaTeX, man-page and texinfo targets) from `cvpack` to `ufedmm`.

### Docs content
- Remove the hand-written autodoc pages in `docs/pythonapi/` and switch the index
  toctree to the generated `api/` pages.
- Keep a short hand-written cvpack page that explains the `ufedmm.cvpack`
  re-export, with every link pointing to `craabreu.github.io/cvpack/latest`.
- `docs/changelog.rst`: include `CHANGELOG.md`, or link to it if the Sphinx setup
  can't parse Markdown.
- Add a `bajpai2023` entry to `docs/refs.bib`, and a "Citing UFEDMM" note on
  `docs/index.rst`.

### Repository files
- `CHANGELOG.md`: add a 0.4.0 entry that covers the new Python and OpenMM
  requirements, the PyPI channel, the release-driven deploys and the new docs
  URL.
- `CITATION.cff`: describe the software, version 0.4.0, MIT license, the
  repository and the docs URL, plus a `preferred-citation` block of type
  `article` for the JCC 2023 paper with all six authors, the journal, volume 44,
  issue 28, pages 2166–2183, year 2023 and doi `10.1002/jcc.27182`.
- `README.md`:
  - Badges: Linter, PyPI version, docs pointing to `/latest`, and coverage
    pointing to `/latest/coverage`.
  - Install instructions for both conda (`-c conda-forge -c mdtools`) and pip.
  - A "Citing UFEDMM" section with the reference and BibTeX.
  - cvpack links pointing to github.com/craabreu/cvpack.
- `ufedmm/_version.py`: set to `0.4.0`.

### Actions for the user
- Add a `PYPI_TOKEN` secret to the repo.
- Check that `ANACONDA_TOKEN` is still valid.
- After merging, create the v0.4.0 GitHub Release.

### Acceptance criteria
- `sphinx-build docs html` runs locally with no new warnings, and the generated
  API pages cover every public class and function.
- Locally, `python -m build` and `twine check dist/*` pass, and `conda build` of
  the recipe succeeds with `UFEDMM_VERSION=v0.4.0`.
- On the PR, the Docs workflow builds but doesn't deploy.
- After merging (with user approval), the manual runs succeed.

## Out of scope

- Any change to ufedmm's algorithms or public API.
- A conda-forge feedstock. As with openxps, it is deferred.
- Converting the examples to notebooks.
