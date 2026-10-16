# Modernize ufedmm Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring ufedmm's packaging, CI, linting, release deployment and docs up to
the current state of openxps, without changing ufedmm's scientific behavior.

**Architecture:** Most ufedmm infrastructure files are replaced by their openxps
counterparts, with `openxps` changed to `ufedmm` and `OPENXPS_VERSION` changed to
`UFEDMM_VERSION`. The work is split into three stacked PRs: (1) config, CI and
packaging; (2) a ruff restyle with no behavior change; (3) release-driven deploys,
PyPI, versioned docs and citation metadata. One real bug that the lint pass
uncovered gets its own PR (Task 7).
That PR repairs `mean_force_free_energy`, which is broken in four ways on current NumPy and OpenMM.

**Tech Stack:** setuptools + versioningit, ruff, pytest/pytest-cov, GitHub Actions
(`mamba-org/setup-micromamba@v2`), conda-build, build/twine, Sphinx (alabaster,
sphinxcontrib-bibtex, sphinx-copybutton).

**Spec:** `docs/superpowers/specs/2026-10-15-modernize-infrastructure-design.md`

## Global Constraints

- Python `>=3.10`. OpenMM `>=8.4`. The CI matrix is exactly openxps's: Linux
  3.14/8.6, 3.13/8.6, 3.12/8.5, 3.11/8.4 and 3.10/8.4, plus macOS 3.13/8.6 and
  Windows 3.13/8.6.
- License: MIT. The license file is `LICENSE.md`.
- Version: `ufedmm/_version.py` is committed and bumped by hand. It stays `0.3.2`
  through PR 2 and becomes `0.4.0` in PR 3.
- Deploys run on `release: published` and on `workflow_dispatch` with a `tag`
  input. Prereleases are skipped.
- Conda channel: `mdtools`. Builds use `-c conda-forge -c mdtools`.
- cvpack links point to `https://github.com/craabreu/cvpack` (project) and
  `https://craabreu.github.io/cvpack/latest/api/<Name>.html` (API).
- Docs URL: `https://craabreu.github.io/ufedmm/latest`.
- Preferred citation: S. Bajpai, B. K. Petkov, M. Tong, C. R. A. Abreu,
  N. N. Nair, M. E. Tuckerman, *J. Comput. Chem.* **44**(28), 2166–2183 (2023),
  doi:10.1002/jcc.27182.
- Work happens in the shared checkout `/data3/craabreu/github_repos/ufedmm`.
  Never `pip install` into, or otherwise modify, the shared env
  `/data3/craabreu/envs/openxps-test`. Throwaway envs go in
  `$SCRATCH=/tmp/claude-1020/-data3-craabreu-github-repos/4f14cf3d-22e6-4ee2-befb-66684ffd3a06/scratchpad`.
- Tools: `T=/data3/craabreu/envs/openxps-test/bin` provides `python`, `pytest` and
  `ruff` 0.17. `/data3/craabreu/envs/condabuild/bin` provides `conda-build`.
- Every commit message ends with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_012Px2sYfcPG4WRLy6RJd3HP
  ```
- Every PR body ends with:
  ```
  🤖 Generated with [Claude Code](https://claude.com/claude-code)

  https://claude.ai/code/session_012Px2sYfcPG4WRLy6RJd3HP
  ```
- Baseline as of 2026-10-15 (the shared env has openmm 8.6.1 and numpy 2.5.4):
  24 tests pass, and 1 doctest fails
  (`ExtendedSpaceState.getDynamicalVariables`, because of the NumPy 2 scalar
  repr). Task 1 fixes it.

## Review Focus

1. **Installing from a fresh wheel.** `AlanineDipeptideModel()` reads
   `ufedmm/data/alanine-dipeptide.pdb`, which must be inside the wheel. If it's
   missing, users hit a FileNotFoundError on first use. Task 2 pins this by
   checking the wheel's contents.
2. **Declared floors are enforced at install time.** A user with Python 3.9 or
   OpenMM 8.1 should get a resolver error, not a runtime break. Task 2 pins
   `Requires-Python: >=3.10` and `Requires-Dist: openmm>=8.4` in the wheel
   metadata.
3. **`ufedmm.cvpack` re-export keeps working.** Scripts call
   `ufedmm.cvpack.Torsion`, and the docs generator must not trip over the
   re-export. Task 1 adds `test_cvpack_reexport`. Task 9 checks that the API pages
   don't contain cvpack classes.
4. **Malformed `workflow_dispatch` tag.** The recipe does
   `environ['UFEDMM_VERSION'][1:]`, so typing `0.4.0` without the `v` would build
   version `.4.0`. Task 8 adds a guard step that fails fast unless the tag
   matches `^v[0-9]+\.[0-9]+\.[0-9]+$`.
5. **gh-pages migration takes the site down.** Deleting the flat pages without
   keeping `.nojekyll` and adding the redirect leaves a 404 at the root URL.
   Task 11 checks the root redirect and `latest/` with curl after the migration.

---

# PR 1: CI, packaging and lint config (branch `modernize-ci-packaging`)

The branch already exists and contains the spec commit `ec70884`. Before Task 1,
run `git -C /data3/craabreu/github_repos/ufedmm switch modernize-ci-packaging`.

### Task 1: Export `__version__`, fix the NumPy 2 doctest, pin the cvpack re-export

**Files:**
- Modify: `ufedmm/__init__.py`
- Modify: `ufedmm/ufedmm.py:835-836` (doctest of `ExtendedSpaceState.getDynamicalVariables`)
- Create: `ufedmm/tests/test_package.py`

**Interfaces:**
- Produces: `ufedmm.__version__: str`. Tasks 2, 9 and 10 rely on it.

- [ ] **Step 1: Write the failing tests**

`ufedmm/tests/test_package.py`:
```python
import re

import ufedmm


def test_version_is_exported():
    assert re.fullmatch(r"\d+\.\d+\.\d+.*", ufedmm.__version__)


def test_cvpack_reexport():
    import cvpack

    import ufedmm.cvpack

    assert ufedmm.cvpack.Torsion is cvpack.Torsion
```

- [ ] **Step 2: Run them and check the failure**

Run: `cd /data3/craabreu/github_repos/ufedmm && $T/pytest --no-cov -p no:cacheprovider ufedmm/tests/test_package.py`
Expected: `test_version_is_exported` FAILS with `AttributeError: module 'ufedmm' has no attribute '__version__'`. `test_cvpack_reexport` PASSES; it pins existing behavior.

- [ ] **Step 3: Export the version**

Add this line to `ufedmm/__init__.py`, directly after the docstring:
```python
from ._version import __version__  # noqa: F401
```

- [ ] **Step 4: Fix the doctest without changing behavior**

In `ufedmm/ufedmm.py`, replace the two doctest lines
```
            >>> context.getState(getPositions=True).getDynamicalVariables()
            [-3.141592653589793, -3.141592653589793]
```
with
```
            >>> state = context.getState(getPositions=True)
            >>> [float(x) for x in state.getDynamicalVariables()]
            [-3.141592653589793, -3.141592653589793]
```

- [ ] **Step 5: Run the whole suite**

Run: `$T/pytest -p no:cacheprovider`
Expected: 27 passed, 0 failed.

- [ ] **Step 6: Commit**

```bash
git add ufedmm/__init__.py ufedmm/ufedmm.py ufedmm/tests/test_package.py
git commit -m "Export __version__ and make the doctest NumPy 2 compatible"   # plus trailer
```

### Task 2: pyproject, setup.cfg, MANIFEST, license and cruft

**Files:**
- Modify: `pyproject.toml`, `MANIFEST.in`
- Create: `setup.cfg`
- Rename: `LICENSE` → `LICENSE.md`
- Delete: `.readthedocs.yml`, `.pylintrc`

**Interfaces:**
- Consumes: `ufedmm.__version__` (Task 1).
- Produces: a `[tool.ruff]` config that Tasks 3, 5 and 6 rely on, including a
  temporary ignore list marked `# TODO(PR2): remove` that Task 6 deletes.

- [ ] **Step 1: Write the failing packaging check**

Create `$SCRATCH/check_wheel.py`:
```python
"""Usage: python -I check_wheel.py <dist_dir>"""
import pathlib
import sys
import zipfile

dist = pathlib.Path(sys.argv[1])
wheel = next(dist.glob("ufedmm-*.whl"))
with zipfile.ZipFile(wheel) as z:
    names = z.namelist()
    meta = z.read(next(n for n in names if n.endswith(".dist-info/METADATA"))).decode()
assert "ufedmm/data/alanine-dipeptide.pdb" in names, "pdb missing from wheel"
assert "Requires-Python: >=3.10" in meta, "python floor missing"
assert "Requires-Dist: openmm>=8.4" in meta, "openmm floor missing"
assert "Requires-Dist: cvpack" in meta, "cvpack dependency missing"
assert "License :: OSI Approved :: MIT License" in meta, "MIT classifier missing"
assert "Proprietary" not in meta, "stale proprietary classifier"
print("wheel OK:", wheel.name)
```
Create a throwaway build env:
`python3 -m venv $SCRATCH/pkgenv && $SCRATCH/pkgenv/bin/pip install -q build twine`

- [ ] **Step 2: Run it against the current tree and check the failure**

Run: `cd /data3/craabreu/github_repos/ufedmm && rm -rf $SCRATCH/dist && $SCRATCH/pkgenv/bin/python -m build -o $SCRATCH/dist . && $SCRATCH/pkgenv/bin/python -I $SCRATCH/check_wheel.py $SCRATCH/dist`
Expected: AssertionError `python floor missing`.

- [ ] **Step 3: Rewrite `pyproject.toml`**

Change the `[project]` table:
```toml
[project]
name = "ufedmm"
description = "Unified Free Energy Dynamics with OpenMM"
dynamic = ["version"]
readme = "README.md"
authors = [
    { name = "Charlles Abreu", email = "craabreu@gmail.com" }
]
license = { text = "MIT" }
# See https://pypi.org/classifiers/
classifiers = [
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Programming Language :: Python :: 3.14",
]
requires-python = ">=3.10"
# Declare any run-time dependencies that should be installed with the package.
dependencies = [
    "cvpack",
    "numpy",
    "openmm>=8.4",
    "pandas",
    "pyyaml",
    "scipy",
]

[project.urls]
"Source" = "https://github.com/craabreu/ufedmm"
"Documentation" = "https://craabreu.github.io/ufedmm/latest"
"Changelog" = "https://github.com/craabreu/ufedmm/blob/main/CHANGELOG.md"

[project.optional-dependencies]
test = [
  "pytest>=7.0",
  "pytest-cov",
]
```
The author email is updated to `craabreu@gmail.com`, the one openxps uses (user request). PR 2 (Task 6) also updates the `.. moduleauthor::` lines in the docstrings.
Keep `[tool.setuptools]`, `[tool.setuptools.packages.find]` and all
`[tool.versioningit*]` tables as they are. Change `[tool.setuptools.package-data]`
to:
```toml
[tool.setuptools.package-data]
ufedmm = [
    "data/alanine-dipeptide.pdb"
]
```
`data/look_and_say.dat` stays unpackaged on purpose: no code reads it. Then
delete the `[tool.isort]` and `[tool.flake8]` tables and add these sections. They
are copied from openxps; only the `ignore` list and the `--cov` target differ:
```toml
[tool.ruff]
line-length = 88
target-version = "py310"
extend-exclude = ["_version.py"]

[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings
    "F",    # pyflakes
    "I",    # isort
    "PL",   # pylint
    "UP",   # pyupgrade
    "B",    # flake8-bugbear
    "C4",   # flake8-comprehensions
]
ignore = [
    "B905",
    # TODO(PR2): remove the rules below after the ruff restyle
    "B006", "B007", "B023", "B904", "C408", "C416", "C417", "E501",
    "I001", "PLR0912", "PLR0913", "PLR0917", "PLR1714", "PLR2004",
    "UP004", "UP015", "UP032",
]

[tool.ruff.lint.pylint]
max-locals = 15
max-args = 6  # Allow up to 6 arguments in function definitions

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401"]  # Allow unused imports in __init__.py
"**/tests/**/*.py" = ["PLR2004"]  # Allow magic values in tests

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
line-ending = "auto"
```
Change the pytest `addopts` to:
```toml
addopts = "-v -ra --doctest-modules --cov=ufedmm --cov-report=term --cov-report=html --cov-report=xml"
```

- [ ] **Step 4: Add `setup.cfg`, fix MANIFEST, rename the license, delete cruft**

`setup.cfg`:
```ini
# Helper file to handle all configs

[coverage:run]
# .coveragerc to control coverage.py and pytest-cov
omit =
    # Omit the tests
    */tests/*
    # Omit generated versioneer
    ufedmm/_version.py

[aliases]
test = pytest
```
`MANIFEST.in`:
```
include LICENSE.md
include MANIFEST.in
include CODE_OF_CONDUCT.md

graft ufedmm
global-exclude *.py[cod] __pycache__ *.so
```
Then run:
```bash
git mv LICENSE LICENSE.md
git rm .readthedocs.yml .pylintrc
```

- [ ] **Step 5: Re-run the packaging check, the lint check and the tests**

Run: `rm -rf $SCRATCH/dist && $SCRATCH/pkgenv/bin/python -m build -o $SCRATCH/dist . && $SCRATCH/pkgenv/bin/python -I $SCRATCH/check_wheel.py $SCRATCH/dist && $SCRATCH/pkgenv/bin/twine check $SCRATCH/dist/*`
Expected: `wheel OK: ufedmm-...whl`, then `PASSED` for both dists.
Run: `$T/ruff check ufedmm && $T/pytest -p no:cacheprovider`
Expected: `All checks passed!`, then 27 passed.
Run: `git status --short ufedmm/_version.py`
Expected: if `python -m build` rewrote the file (versioningit), restore it with `git checkout ufedmm/_version.py` so it still reads `__version__ = "0.3.2"`.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml setup.cfg MANIFEST.in
git commit -m "Modernize packaging metadata and switch lint config to ruff"   # plus trailer
```

### Task 3: Conda envs, recipe, CI workflows and contributor tooling

**Files:**
- Modify: `devtools/conda-envs/test_env.yaml`, `lint_env.yaml`, `docs_env.yaml`
- Delete: `devtools/conda-envs/conda_build_env.yaml`
- Create: `devtools/conda-envs/deployment_env.yaml`
- Modify: `devtools/conda-recipes/anaconda/meta.yaml`
- Modify: `.github/workflows/Linux.yaml`, `MacOS.yaml`, `Windows.yaml`, `Linter.yaml`
- Modify: `devtools/scripts/format_and_check.sh`, `devtools/README.md`, `docs/contributing.rst`

**Interfaces:**
- Consumes: the ruff config from Task 2.
- Produces: `deployment_env.yaml`, which Task 8 uses.

- [ ] **Step 1: Write the failing config check**

Create `$SCRATCH/check_ci.py`:
```python
"""Usage: python -I check_ci.py <repo>"""
import pathlib
import sys

import yaml

repo = pathlib.Path(sys.argv[1])
wf = repo / ".github" / "workflows"
for name in ["Linux", "MacOS", "Windows", "Linter"]:
    text = (wf / f"{name}.yaml").read_text()
    yaml.safe_load(text)
    assert "provision-with-micromamba" not in text, name
    assert "setup-micromamba@v2" in text, name
    assert "actions/checkout@v4" in text, name
linux = yaml.safe_load((wf / "Linux.yaml").read_text())
rows = {(r["python-version"], r["openmm-version"])
        for r in linux["jobs"]["test"]["strategy"]["matrix"]["include"]}
assert rows == {("3.14", "8.6"), ("3.13", "8.6"), ("3.12", "8.5"),
                ("3.11", "8.4"), ("3.10", "8.4")}, rows
assert "ruff check ufedmm" in (wf / "Linter.yaml").read_text()
envs = repo / "devtools" / "conda-envs"
assert not (envs / "conda_build_env.yaml").exists()
deploy = yaml.safe_load((envs / "deployment_env.yaml").read_text())
assert "boa" not in str(deploy) and "twine" in str(deploy)
assert "ruff" in (envs / "lint_env.yaml").read_text()
for f in envs.glob("*.yaml"):
    yaml.safe_load(f.read_text())
print("CI config OK")
```

- [ ] **Step 2: Run it and check the failure**

Run: `$T/python -I $SCRATCH/check_ci.py /data3/craabreu/github_repos/ufedmm`
Expected: AssertionError `Linux`.

- [ ] **Step 3: Port the test and lint workflows from openxps**

```bash
cd /data3/craabreu/github_repos/ufedmm
for w in Linux MacOS Windows Linter; do
  sed 's/openxps/ufedmm/g' ../openxps/.github/workflows/$w.yaml > .github/workflows/$w.yaml
done
```
Check the result with `git diff .github/workflows`. `Linter.yaml` must end in `run: ruff check ufedmm`.

- [ ] **Step 4: Rewrite the conda envs**

`devtools/conda-envs/test_env.yaml`:
```yaml
name: test
channels:
  - conda-forge
  - mdtools
dependencies:
    # Base depends
  - python
  - pip
  - numpy
  - openmm
  - pandas
  - pyyaml
  - scipy
  - cvpack

    # Testing
  - pytest
  - pytest-cov
```
`devtools/conda-envs/lint_env.yaml`:
```yaml
name: linter
channels:
  - conda-forge
  - mdtools
dependencies:
    # Base depends
  - python
  - pip
  - numpy
  - openmm
  - pandas
  - pyyaml
  - scipy
  - cvpack

    # Linting
  - ruff
```
`devtools/conda-envs/docs_env.yaml`:
```yaml
name: docs
channels:
  - conda-forge
  - mdtools
dependencies:
    # Base depends
  - python
  - pip
  - numpy
  - openmm
  - pandas
  - pyyaml
  - scipy
  - cvpack

    # Testing
  - pytest
  - pytest-cov

    # Docs building
  - sphinx
  - sphinxcontrib-bibtex
  - sphinx-copybutton
```
`devtools/conda-envs/deployment_env.yaml`:
```yaml
name: build
channels:
  - conda-forge
  - mdtools
dependencies:
  # Base dependencies
  - python
  - pip
  - numpy
  - openmm
  - pandas
  - pyyaml
  - scipy
  - cvpack

  # Conda build
  - anaconda-client
  - conda-build

  # PyPI build
  - pip:
    - build
    - twine
```
Then run `git rm devtools/conda-envs/conda_build_env.yaml`.

- [ ] **Step 5: Update the conda recipe**

In `devtools/conda-recipes/anaconda/meta.yaml`, replace the `run:` list with:
```yaml
  run:
    - python >=3.10
    - numpy
    - openmm >=8.4
    - pandas
    - pyyaml
    - scipy
    - cvpack
```
Also set `license_file: LICENSE.md` and `doc_url: https://craabreu.github.io/ufedmm/latest`.
Leave the existing tag-push `Anaconda.yaml` workflow alone in this PR; Task 8
replaces it. Its only reference was to `conda_build_env.yaml`, so in
`.github/workflows/Anaconda.yaml` change `conda_build_env.yaml` to
`deployment_env.yaml`, `conda mambabuild` to `conda build`, and `-name *.tar.bz2`
to `\( -name '*.conda' -o -name '*.tar.bz2' \)`. That keeps it working until
Task 8.

- [ ] **Step 6: Update the contributor tooling**

`devtools/scripts/format_and_check.sh`:
```bash
#!/usr/bin/env bash
set -e -v
ruff format ufedmm
ruff check --fix ufedmm
```
In `devtools/README.md`, replace every mention of `conda_build_env.yaml` with
`deployment_env.yaml`. Replace every mention of flake8, black or isort with ruff
(`grep -n -i -E 'flake8|black|isort|conda_build' devtools/README.md` lists them).
In `docs/contributing.rst`:
- Replace the four-item prerequisites list (lines 39–45) with
  `1. pytest and pytest-cov` and `2. ruff`.
- Replace the two-line "Tips" code block with `devtools/scripts/format_and_check.sh`.
- Replace `devtools/run_tests.sh` with `pytest`, in both places.
- Delete the footnote `[1]` about Travis, and the `[1]_` reference to it.

- [ ] **Step 7: Run the checks**

Run: `$T/python -I $SCRATCH/check_ci.py /data3/craabreu/github_repos/ufedmm`
Expected: `CI config OK`.
Run: `cd devtools/conda-recipes/anaconda && UFEDMM_VERSION=v0.3.2 /data3/craabreu/envs/condabuild/bin/conda-render . -c conda-forge -c mdtools | grep -E 'version|openmm|python'`
Expected: `version: 0.3.2`, `openmm >=8.4`, `python >=3.10`.
Run: `cd /data3/craabreu/github_repos/ufedmm && bash -n devtools/scripts/format_and_check.sh && $T/ruff check ufedmm && $T/pytest -p no:cacheprovider -q`
Expected: no syntax error, `All checks passed!`, 27 passed.

- [ ] **Step 8: Commit**

```bash
git add -A .github devtools docs/contributing.rst
git commit -m "Port CI workflows and conda environments from openxps"   # plus trailer
```

### Task 4: Open PR 1

- [ ] **Step 1: Push and open the PR**

```bash
git push -u origin modernize-ci-packaging
gh pr create --base main --title "Modernize CI, packaging and lint configuration" --body "<summary: what changed (Tasks 1–3), the new Python>=3.10 / OpenMM>=8.4 floor, the temporary ruff ignores removed in the follow-up restyle PR, local verification results; then the PR trailer>"
```
- [ ] **Step 2: Watch CI**

Run: `gh pr checks --watch`
Expected: Linux (5 rows), MacOS, Windows and Linter are green. Docs may stay
red or flaky until PR 3, because it still uses `REPO_TOKEN`; mention this in the
PR body. If a test row fails, use superpowers:systematic-debugging before
changing anything.

---

# PR 2: ruff restyle (branch `ruff-restyle`)

Create the branch from `main` after PR 1 merges. If PR 1 hasn't merged yet,
create it from `modernize-ci-packaging` and open the PR against that branch.
Record the baseline first:
`$T/pytest -p no:cacheprovider -q 2>&1 | tail -1 > $SCRATCH/baseline.txt`
It should say 27 passed.

### Task 5: Mechanical ruff format

**Files:**
- Modify: every `*.py` file under `ufedmm/` and `examples/` (formatter only)
- Create: `.git-blame-ignore-revs`

- [ ] **Step 1: Show that the formatter check fails**

Run: `$T/ruff format --check ufedmm examples`
Expected: exit 1 and `7 files would be reformatted` (the count is approximate).

- [ ] **Step 2: Format and apply the safe autofixes**

```bash
$T/ruff format ufedmm examples
$T/ruff check --fix --select I001,UP004,UP015,UP032 ufedmm examples
```
Don't touch `examples/RESPA/RESPA_tutorial.ipynb`. If ruff changed it, run
`git checkout examples/RESPA/RESPA_tutorial.ipynb`.

- [ ] **Step 3: Check that the tests are unchanged**

Run: `$T/pytest -p no:cacheprovider -q 2>&1 | tail -1 | diff - $SCRATCH/baseline.txt` (the timing in the line may differ; compare the counts)
Expected: the same pass and fail counts as the baseline.

- [ ] **Step 4: Commit, then record the SHA**

```bash
git commit -am "Apply ruff format and safe autofixes"   # plus trailer
printf '# ruff format (PR 2)\n%s\n' "$(git rev-parse HEAD)" > .git-blame-ignore-revs
git add .git-blame-ignore-revs
git commit -m "Ignore the ruff restyle commit in git blame"   # plus trailer
```

### Task 6: Hand-fix the remaining lint and remove the temporary ignores

**Files:**
- Modify: `pyproject.toml` (remove the `TODO(PR2)` block)
- Modify: `ufedmm/analysis.py`, `ufedmm/integrators.py`, `ufedmm/io.py`, `ufedmm/ufedmm.py`, `ufedmm/tests/test_ufedmm.py`

- [ ] **Step 1: Remove the temporary ignores and look at the failures**

In `pyproject.toml`, set `ignore = ["B905"]`, deleting the `TODO(PR2)` lines.
Run: `$T/ruff check --output-format concise ufedmm`
Expected: about 35 violations, all from the rules listed in Step 2.

- [ ] **Step 2: Fix each rule like this**

| Rule | Fix |
|------|-----|
| B006 (`analysis.py` `properties={}`; `integrators.py` `respa_loops=[1]` and `embodied_force_groups=[]`) | Default to `None` and add `if x is None: x = <old literal>` as the first statement of the body. Keep the docstring's `default=` text. Neither value is ever mutated, so this doesn't change behavior. |
| B904 (`integrators.py:67`, `io.py:173`) | Write `except StopIteration:` → `raise Exception(...) from None`. |
| B007 | Rename unused loop variables to `_`. Inside `enumerate(...)`, rename only the unused element. |
| C408, C416, C417, PLR1714 | Use the rewrite ruff suggests: a literal, `list(x)`, a generator expression, or `not in (sys.stdout, sys.stderr)`. |
| E501 (`ufedmm.py:916`) | Split the long string into implicitly concatenated pieces. |
| PLR0913, PLR0917 (public `__init__` and method signatures) | Add `# noqa: PLR0913, PLR0917` to the `def` line. Signatures are public API. |
| PLR0912 (`ufedmm.py:29`, `io.py:229`) | Add `# noqa: PLR0912` to the `def` line. Refactoring isn't in scope. |
| PLR2004 outside tests | Add `# noqa: PLR2004` to the line. |
| Email (`.. moduleauthor::` in every module docstring) | Replace `abreu@eq.ufrj.br` with `craabreu@gmail.com`. |
| B023 (`analysis.py:211-216`) | **Don't fix this in this PR.** Add `# noqa: B023` to each lambda line, plus the comment `# late-binding bug, fixed in the fix-mean-force-free-energy PR` above the loop. Task 7 fixes it with a regression test. |

- [ ] **Step 3: Check that lint is clean and tests are unchanged**

Run: `$T/ruff check ufedmm && $T/ruff format --check ufedmm examples`
Expected: `All checks passed!`, then `N files already formatted`.
Run: `$T/pytest -p no:cacheprovider -q 2>&1 | tail -1`
Expected: the same counts as `$SCRATCH/baseline.txt`.

- [ ] **Step 4: Commit and open PR 2**

```bash
git commit -am "Fix remaining ruff violations and drop temporary ignores"   # plus trailer
git push -u origin ruff-restyle
gh pr create --base main --title "Restyle the package with ruff" --body "<summary: formatting-only first commit (in .git-blame-ignore-revs), hand fixes with no behavior change, test counts before and after, B023 noted for the follow-up fix PR; PR trailer>"
gh pr checks --watch
```
Expected: all checks green.

### Task 7: Repair `mean_force_free_energy` (branch `fix-mean-force-free-energy`, separate PR)

Approved by the user on 2026-10-15. Branch it from `main` once PR 2 has merged.

**Background, verified 2026-10-15 in a scratch copy:**
`FreeEnergyAnalyzer.mean_force_free_energy` is broken on current NumPy and
OpenMM in four separate ways. No existing test calls it.
1. `kernel` calls `np.sum(<generator>)`, which NumPy 2.x rejects with a TypeError.
2. The lambdas in the `exponent` and `derivative` lists capture `factor` and
   `variance` late (ruff B023). With several CVs, the least-squares fit uses the
   last CV's width and period for every axis, so the fitted weights are wrong.
   The OpenMM `_RBFContext` that evaluates the result *does* use per-CV widths,
   so the result is internally inconsistent.
3. `mean_force` reads `state.getEnergyParameterDerivatives()._value`. On OpenMM
   8.x that returns a `mapstringdouble`, which has no `_value`.
4. `np.vectorize(mean_force)` can't return a vector per call. It needs a
   `signature`.

The regression test below fails with any one of these bugs present and passes
once all four are fixed. Both outcomes were checked.

**Files:**
- Modify: `ufedmm/analysis.py` (inside `mean_force_free_energy`, around lines 207–252)
- Create: `ufedmm/tests/test_analysis.py`

**Interfaces:**
- Consumes: `FreeEnergyAnalyzer(ufed, dataframe)` and
  `mean_force_free_energy(centers, mean_forces, sigma, platform_name="Reference", properties=None)`,
  which returns `(potential, mean_force)`. Both returned functions are
  vectorized. `mean_force(*x)` returns an array of length `n`, one entry per
  variable.

- [ ] **Step 1: Write the failing regression test**

`ufedmm/tests/test_analysis.py`:
```python
import itertools

import numpy as np

import ufedmm


def test_mean_force_fit_uses_each_variables_own_sigma():
    model = ufedmm.AlanineDipeptideModel()
    sigmas = [0.3, 0.6]
    s_phi = ufedmm.DynamicalVariable(
        "s_phi", -3, 3, 50, 300, model.phi, 1000, periodic=False
    )
    s_psi = ufedmm.DynamicalVariable(
        "s_psi", -3, 3, 50, 300, model.psi, 1000, periodic=False
    )
    ufed = ufedmm.UnifiedFreeEnergyDynamics([s_phi, s_psi], 300)
    analyzer = ufedmm.FreeEnergyAnalyzer(ufed, dataframe=None)

    # Mean forces generated from a known Gaussian RBF expansion on the same grid
    # must be reproduced exactly by the fitted model.
    grid = np.array(list(itertools.product([-0.5, 0.0, 0.5], repeat=2)))
    weights = np.random.default_rng(0).normal(size=len(grid))
    variances = np.square(sigmas)

    def true_force(x):
        d = x - grid
        k = weights * np.exp(-0.5 * np.sum(d**2 / variances, axis=1))
        return np.sum(k[:, None] * d / variances, axis=0)

    forces = np.array([true_force(x) for x in grid])
    centers = [grid[:, 0], grid[:, 1]]
    mean_forces = [forces[:, 0], forces[:, 1]]
    _, mean_force = analyzer.mean_force_free_energy(centers, mean_forces, sigmas)
    fitted = np.array([np.asarray(mean_force(*x), dtype=float) for x in grid])
    assert np.allclose(fitted, forces, atol=1e-6)
```

- [ ] **Step 2: Run it and check the failure**

Run: `$T/pytest --no-cov -p no:cacheprovider ufedmm/tests/test_analysis.py`
Expected: FAIL with `TypeError: Calling np.sum(generator) is deprecated` (bug 1
shows up first).

- [ ] **Step 3: Fix all four bugs**

In `ufedmm/analysis.py`, inside `mean_force_free_energy`:

(a) Bind the loop variables early (bug 2). Replace the loop with:
```python
        for v, variance in zip(variables, variances):
            if v.periodic:  # von Mises
                factor = 2 * np.pi / v._range
                exponent.append(
                    lambda x, f=factor, var=variance: (np.cos(f * x) - 1.0)
                    / (f * f * var)
                )
                derivative.append(
                    lambda x, f=factor, var=variance: -np.sin(f * x) / (f * var)
                )
            else:  # Gauss
                exponent.append(lambda x, var=variance: -0.5 * x**2 / var)
                derivative.append(lambda x, var=variance: -x / var)
```
Also remove the `# noqa: B023` markers and the comment that Task 6 added.

(b) Replace `np.sum(...)` with the built-in `sum(...)` (bug 1):
```python
        def kernel(x):
            return np.exp(sum(exponent[i](x[i]) for i in range(n)))
```

(c) Read the derivatives by name (bug 3). Replace
`return -state.getEnergyParameterDerivatives()._value` with:
```python
            derivatives = state.getEnergyParameterDerivatives()
            return -np.array([derivatives[p] for p in context.parameters])
```

(d) Give the vectorized mean force a signature (bug 4). Replace the final
`return np.vectorize(potential), np.vectorize(mean_force)` with:
```python
        signature = ",".join(["()"] * n) + "->(n)"
        return np.vectorize(potential), np.vectorize(mean_force, signature=signature)
```

- [ ] **Step 4: Run the tests and lint**

Run: `$T/pytest -p no:cacheprovider -q && $T/ruff check ufedmm && $T/ruff format --check ufedmm`
Expected: all tests pass, including the new one, and lint is clean.
Run: `cd examples/alanine_dipeptide && grep -n mean_force_free_energy analysis.py`.
If the example calls it, check that the way the example uses the outputs still
works with the vector-valued `mean_force`. Read the example; don't run the full
analysis.

- [ ] **Step 5: Commit and open the PR**

```bash
git add ufedmm/analysis.py ufedmm/tests/test_analysis.py
git commit -m "Repair mean_force_free_energy on current NumPy and OpenMM"   # plus trailer
git push -u origin fix-mean-force-free-energy
gh pr create --base main --title "Repair mean_force_free_energy and fix per-variable kernel widths" --body "<the four bugs, who was affected (any multi-CV analysis with distinct sigmas or periodicities, and every call on NumPy 2 or OpenMM 8), the regression test; PR trailer>"
```
Add the `### Fixed` entry for this to the CHANGELOG 0.4.0 entry in Task 10.

---

# PR 3: release-driven deploys, PyPI and docs (branch `release-deploys-docs`)

Create the branch from `main` after PR 2 merges.

### Task 8: Docs, Anaconda and PyPI deploy workflows

**Files:**
- Modify: `.github/workflows/Docs.yaml`, `.github/workflows/Anaconda.yaml`
- Create: `.github/workflows/PyPI.yaml`

**Interfaces:**
- Consumes: `deployment_env.yaml` and `docs_env.yaml` (Task 3), and the `UFEDMM_VERSION` env var that `meta.yaml` reads.
- Produces: the `UFEDMM_VERSION` env var during docs builds, which Task 9's `conf.py` reads.

- [ ] **Step 1: Extend `check_ci.py` with the deploy checks (they should fail)**

Append to `$SCRATCH/check_ci.py`:
```python
for name in ["Docs", "Anaconda", "PyPI"]:
    text = (wf / f"{name}.yaml").read_text()
    on = yaml.safe_load(text)[True]  # PyYAML parses the key `on` as True
    assert on.get("release") == {"types": ["published"]}, name
    assert "tag" in on["workflow_dispatch"]["inputs"], name
    assert "prerelease" in text and "UFEDMM_VERSION" in text, name
    assert "OPENXPS" not in text and "openxps" not in text, name
for name in ["Anaconda", "PyPI"]:
    assert "^v[0-9]+\\.[0-9]+\\.[0-9]+$" in (wf / f"{name}.yaml").read_text(), name
assert "REPO_TOKEN" not in (wf / "Docs.yaml").read_text()
print("deploy config OK")
```
Run: `$T/python -I $SCRATCH/check_ci.py /data3/craabreu/github_repos/ufedmm`
Expected: it fails on `Docs` (or with `FileNotFoundError` for `PyPI.yaml`).

- [ ] **Step 2: Port the three workflows**

```bash
cd /data3/craabreu/github_repos/ufedmm
for w in Docs Anaconda PyPI; do
  sed -e 's/OPENXPS_VERSION/UFEDMM_VERSION/g' -e 's/openxps/ufedmm/g' \
    ../openxps/.github/workflows/$w.yaml > .github/workflows/$w.yaml
done
```

- [ ] **Step 3: Add the tag guard (Review Focus #4)**

In both `Anaconda.yaml` and `PyPI.yaml`, insert this step right after
`actions/checkout@v4`:
```yaml
      - name: Validate release tag
        shell: bash
        run: |
          [[ "$TAG" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "Invalid tag '$TAG' (expected vX.Y.Z)"; exit 1; }
        env:
          TAG: ${{ inputs.tag || github.event.release.tag_name }}
```

- [ ] **Step 4: Check the ported files**

Run: `$T/python -I $SCRATCH/check_ci.py /data3/craabreu/github_repos/ufedmm`
Expected: `CI config OK`, then `deploy config OK`.
Run: `git diff .github/workflows/Docs.yaml`
Check by eye: the Docs build uses `docs_env.yaml` with `python=3.13 openmm=8.6`,
runs `pytest` before `sphinx-build`, deploys to `docs/development`, or to
`docs/$RELEASE_TAG` and `docs/latest`, and checks out with
`token: ${{ secrets.GITHUB_TOKEN }}`.
Run the guard by hand:
`TAG=0.4.0 bash -c '[[ "$TAG" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] && echo ok || echo rejected'` → `rejected`.
Then run it with `TAG=v0.4.0` → `ok`.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows
git commit -m "Deploy docs, conda and PyPI packages on published releases"   # plus trailer
```

### Task 9: Generated API docs and `conf.py` modernization

**Files:**
- Modify: `docs/conf.py`, `docs/index.rst`
- Delete: `docs/pythonapi/{analysis,integrators,io,testmodels,ufedmm,index}.rst`
- Move: `docs/pythonapi/cvpack.rst` → `docs/cvpack.rst` (rewrite its links)
- Create: `docs/api/.gitignore`

**Interfaces:**
- Consumes: `ufedmm.__version__` (Task 1) and the `UFEDMM_VERSION` env var (Task 8).
- Produces: `docs/api/index.rst` and one `.rst` page per public name, generated at build time.

- [ ] **Step 1: Create the docs env and record the warning baseline**

```bash
conda env create -q -p $SCRATCH/docsenv -f /data3/craabreu/github_repos/ufedmm/devtools/conda-envs/docs_env.yaml python=3.13 openmm=8.6
cd /data3/craabreu/github_repos/ufedmm
$SCRATCH/docsenv/bin/sphinx-build -q docs $SCRATCH/html-before 2> $SCRATCH/warn-before.txt; wc -l $SCRATCH/warn-before.txt
```
Don't `pip install` ufedmm into the docs env. `conf.py` puts the repo on
`sys.path`.

- [ ] **Step 2: Write the failing docs check**

`$SCRATCH/check_docs.py`:
```python
"""Usage: python -I check_docs.py <html_dir>"""
import pathlib
import sys

html = pathlib.Path(sys.argv[1])
expected = [
    "CollectiveVariable", "DynamicalVariable", "ExtendedSpaceContext",
    "ExtendedSpaceSimulation", "UnifiedFreeEnergyDynamics",
    "CustomIntegrator", "GeodesicLangevinIntegrator", "MiddleMassiveNHCIntegrator",
    "MiddleMassiveGGMTIntegrator", "RegulatedNHLIntegrator",
    "AbstractMiddleRespaIntegrator", "add_inner_nonbonded_force",
    "Tee", "StateDataReporter", "serialize", "deserialize",
    "FreeEnergyAnalyzer", "Analyzer", "AlanineDipeptideModel",
]
missing = [n for n in expected if not (html / "api" / f"{n}.html").exists()]
assert not missing, f"missing API pages: {missing}"
assert not (html / "api" / "Torsion.html").exists(), "cvpack leaked into API"
cv = (html / "cvpack.html").read_text()
assert "readthedocs" not in cv and "craabreu.github.io/cvpack/latest" in cv
print("docs OK")
```
Run: `$SCRATCH/docsenv/bin/python -I $SCRATCH/check_docs.py $SCRATCH/html-before`
Expected: AssertionError `missing API pages: [...]`.

- [ ] **Step 3: Rewrite the top of `docs/conf.py`**

Replace everything from `import inspect` down to the end of the
"Project information" block with the code below. `sys.path` is set **before**
`import ufedmm`, so the docs build works without installing the package (the
openxps original imports first). The generator filters names by `__module__`,
because ufedmm's modules don't define `__all__`:
```python
import inspect
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

import ufedmm  # noqa: E402

os.makedirs("api", exist_ok=True)


def _public_members(module):
    return sorted(
        (
            obj
            for name, obj in vars(module).items()
            if not name.startswith("_")
            and (inspect.isclass(obj) or inspect.isfunction(obj))
            and obj.__module__ == module.__name__
        ),
        key=lambda obj: obj.__name__,
    )


def create_class_rst_file(cls):
    name = cls.__name__
    methods = dict(cls.__dict__)
    included_methods = [
        f"    .. automethod:: {method}\n"
        for method in sorted(methods)
        if not method.startswith("_")
        and (callable(getattr(cls, method)) or isinstance(methods[method], property))
    ]
    with open(f"api/{name}.rst", "w") as f:
        f.writelines(
            [
                f"{name}\n",
                "=" * len(name) + "\n\n",
                ".. currentmodule:: ufedmm\n",
                f".. autoclass:: {name}\n",
                "    :member-order: alphabetical\n\n",
            ]
            + ["    .. rubric:: Methods\n\n"] * bool(included_methods)
            + included_methods
        )


def create_function_rst_file(func):
    name = func.__name__
    with open(f"api/{name}.rst", "w") as f:
        f.writelines(
            [
                f"{name}\n",
                "=" * len(name) + "\n\n",
                ".. currentmodule:: ufedmm\n",
                f".. autofunction:: {name}\n",
            ]
        )


def create_module_docs(title, members):
    toctree = f"{title.lower()}.rst"
    with open(f"api/{toctree}", "w") as f:
        f.write(f"{title}\n{'=' * len(title)}\n\n.. toctree::\n    :titlesonly:\n\n")
        for obj in members:
            f.write(f"    {obj.__name__}\n")
            if inspect.isclass(obj):
                create_class_rst_file(obj)
            else:
                create_function_rst_file(obj)
        f.write("\n.. testsetup::\n\n    from ufedmm import *")
    return toctree


_core = [
    ufedmm.CollectiveVariable,
    ufedmm.DynamicalVariable,
    ufedmm.ExtendedSpaceContext,
    ufedmm.ExtendedSpaceSimulation,
    ufedmm.UnifiedFreeEnergyDynamics,
]
_toctrees = [create_module_docs("Core", _core)] + [
    create_module_docs(title, _public_members(module))
    for title, module in [
        ("Integrators", ufedmm.integrators),
        ("IO", ufedmm.io),
        ("Analysis", ufedmm.analysis),
        ("Testmodels", ufedmm.testmodels),
    ]
]

with open("api/index.rst", "w") as f:
    f.write(
        "API Reference\n"
        "=============\n"
        "\n"
        ".. toctree::\n"
        "    :maxdepth: 2\n"
        "    :titlesonly:\n"
        "\n" + "".join(f"    {t}\n" for t in _toctrees)
    )

# -- Project information -----------------------------------------------------

version = os.getenv("UFEDMM_VERSION", ufedmm.__version__)
project = f"UFEDMM {version}"
copyright = r"2020-2026 C. Abreu"
author = "Charlles Abreu"
release = ""
```
Then make these edits further down in `conf.py`:
- In `html_theme_options`, set `"github_repo": "ufedmm"`.
- Set `htmlhelp_basename = "ufedmmdoc"`.
- In `latex_documents`, `man_pages` and `texinfo_documents`, change every
  `"cvpack"` target or name to `"ufedmm"`, and every
  `"Collective Variable Package Documentation"` title to `"UFEDMM Documentation"`.
- Replace `extlinks` with the following. Keep the existing `OpenMM` URL pattern
  if it differs; check with `grep -n -A8 extlinks docs/conf.py`.
```python
extlinks = {
    "OpenMM": (
        "http://docs.openmm.org/latest/api-python/generated/openmm.openmm.%s.html",
        "openmm.%s",
    ),
    "CVPack": (
        "https://craabreu.github.io/cvpack/latest/api/%s.html",
        "cvpack.%s",
    ),
}
```

- [ ] **Step 4: Restructure the docs pages**

```bash
git mv docs/pythonapi/cvpack.rst docs/cvpack.rst
git rm docs/pythonapi/{analysis,integrators,io,testmodels,ufedmm,index}.rst
printf '*\n!.gitignore\n' > docs/api/.gitignore
sed -i 's#https://cvpack.readthedocs.io/en/latest/api/#https://craabreu.github.io/cvpack/latest/api/#g; s#https://cvpack.readthedocs.io/en/latest/#https://github.com/craabreu/cvpack#g' docs/cvpack.rst
```
Check that the class names linked from `docs/cvpack.rst` exist in cvpack:
```bash
grep -o 'api/[A-Za-z]*\.html' docs/cvpack.rst | sed 's#api/##; s#\.html##' | while read n; do $T/python -c "import cvpack, sys; sys.exit(0 if hasattr(cvpack, '$n') else 1)" || echo "MISSING $n"; done
```
Drop any bullet that prints `MISSING`. In `docs/index.rst`, replace
`pythonapi/index` in the toctree with `api/index` followed by `cvpack`.

- [ ] **Step 5: Build and check**

Run: `rm -rf $SCRATCH/html && $SCRATCH/docsenv/bin/sphinx-build -q docs $SCRATCH/html 2> $SCRATCH/warn-after.txt; $SCRATCH/docsenv/bin/python -I $SCRATCH/check_docs.py $SCRATCH/html`
Expected: `docs OK`.
Run: `diff <(sed 's/:[0-9]*:/:/' $SCRATCH/warn-before.txt | sort) <(sed 's/:[0-9]*:/:/' $SCRATCH/warn-after.txt | sort) | grep '^>'`
Expected: no new warnings. Warnings from the removed `pythonapi` pages may
disappear, which is fine. Fix every new warning before committing.
Run: `UFEDMM_VERSION=v9.9.9 $SCRATCH/docsenv/bin/sphinx-build -q docs $SCRATCH/html-v 2>/dev/null; grep -o 'UFEDMM v9.9.9' $SCRATCH/html-v/index.html | head -1`
Expected: `UFEDMM v9.9.9`.

- [ ] **Step 6: Commit**

```bash
git add -A docs
git commit -m "Generate API reference pages and fix template leftovers in conf.py"   # plus trailer
```

### Task 10: Changelog, citation, README and version bump

**Files:**
- Create: `CHANGELOG.md`, `CITATION.cff`
- Modify: `docs/refs.bib`, `docs/index.rst`, `docs/changelog.rst`, `docs/installation.rst`, `README.md`, `ufedmm/_version.py`

- [ ] **Step 1: Write the failing metadata check**

`$SCRATCH/check_meta.py`:
```python
"""Usage: python -I check_meta.py <repo>"""
import pathlib
import re
import sys

import yaml

repo = pathlib.Path(sys.argv[1])
cff = yaml.safe_load((repo / "CITATION.cff").read_text())
pc = cff["preferred-citation"]
assert pc["type"] == "article" and pc["doi"] == "10.1002/jcc.27182"
assert [a["family-names"] for a in pc["authors"]] == [
    "Bajpai", "Petkov", "Tong", "Abreu", "Nair", "Tuckerman"]
assert (pc["volume"], pc["issue"], pc["start"], pc["end"], pc["year"]) == (44, 28, 2166, 2183, 2023)
assert cff["version"] == "0.4.0" and cff["license"] == "MIT"
assert re.search(r'__version__ = "0\.4\.0"', (repo / "ufedmm/_version.py").read_text())
assert "## 0.4.0" in (repo / "CHANGELOG.md").read_text()
readme = (repo / "README.md").read_text()
assert "10.1002/jcc.27182" in readme and "pip install ufedmm" in readme
assert "cvpack.readthedocs" not in readme and "github.com/craabreu/cvpack" in readme
assert "craabreu.github.io/ufedmm/latest" in readme
assert "bajpai2023" in (repo / "docs/refs.bib").read_text()
print("metadata OK")
```
Run: `$T/python -I $SCRATCH/check_meta.py /data3/craabreu/github_repos/ufedmm`
Expected: `FileNotFoundError` for `CITATION.cff`.

- [ ] **Step 2: Write `CITATION.cff`**

```yaml
cff-version: 1.2.0
message: "If you use UFEDMM in published work, please cite the article below."
type: software
title: "UFEDMM: Unified Free Energy Dynamics with OpenMM"
abstract: >-
  UFEDMM extends OpenMM's Python API to run extended phase-space simulations,
  such as TAMD, d-AFED, UFED and extended-space metadynamics, and to compute
  free-energy surfaces as functions of collective variables.
authors:
  - family-names: Abreu
    given-names: Charlles
    orcid: "https://orcid.org/0000-0002-6956-4045"
version: 0.4.0
date-released: 2026-10-15
license: MIT
repository-code: "https://github.com/craabreu/ufedmm"
url: "https://craabreu.github.io/ufedmm/latest"
keywords:
  - molecular dynamics
  - enhanced sampling
  - extended phase space
  - free energy
  - OpenMM
preferred-citation:
  type: article
  title: >-
    An interoperable implementation of collective-variable based enhanced
    sampling methods in extended phase space within the OpenMM package
  authors:
    - family-names: Bajpai
      given-names: Shitanshu
    - family-names: Petkov
      given-names: Brian K.
    - family-names: Tong
      given-names: Muchen
    - family-names: Abreu
      given-names: Charlles R. A.
      orcid: "https://orcid.org/0000-0002-6956-4045"
    - family-names: Nair
      given-names: Nisanth N.
    - family-names: Tuckerman
      given-names: Mark E.
  journal: "Journal of Computational Chemistry"
  volume: 44
  issue: 28
  start: 2166
  end: 2183
  year: 2023
  doi: "10.1002/jcc.27182"
```
Set `date-released` to the actual release date when the user publishes the
release, if that isn't 2026-10-15.

- [ ] **Step 3: Write `CHANGELOG.md` and bump the version**

`CHANGELOG.md`:
```markdown
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
```
If Task 7 merged, add:
```markdown
### Fixed
- `FreeEnergyAnalyzer.mean_force_free_energy` works again with NumPy 2 and
  OpenMM 8, and its fit now uses each collective variable's own kernel width and
  period. Before this fix, every kernel used the last variable's values.
```
Set `ufedmm/_version.py` to `__version__ = "0.4.0"`.
Replace the body of `docs/changelog.rst` (keep the title) with:
```rst
The full changelog is kept in
`CHANGELOG.md <https://github.com/craabreu/ufedmm/blob/main/CHANGELOG.md>`_.
```

- [ ] **Step 4: Add the citation to the docs and the README**

Append to `docs/refs.bib`:
```bibtex
@article{bajpai2023,
  author  = {Bajpai, Shitanshu and Petkov, Brian K. and Tong, Muchen and Abreu, Charlles R. A. and Nair, Nisanth N. and Tuckerman, Mark E.},
  title   = {An interoperable implementation of collective-variable based enhanced sampling methods in extended phase space within the {OpenMM} package},
  journal = {Journal of Computational Chemistry},
  volume  = {44},
  number  = {28},
  pages   = {2166--2183},
  year    = {2023},
  doi     = {10.1002/jcc.27182},
}
```
In `docs/index.rst`, insert this before `Contents`:
```rst
Citing UFEDMM
=============

If you use UFEDMM in published work, please cite :cite:`bajpai2023`.
```
In `README.md`:
1. Replace the badge block with these badges, in this order: Linux, MacOS,
   Windows, Linter
   (`https://github.com/craabreu/ufedmm/workflows/Linter/badge.svg`), Docs
   linking to `https://craabreu.github.io/ufedmm/latest`, Coverage
   (`https://craabreu.github.io/ufedmm/latest/coverage/coverage.svg`, linking to
   `.../latest/coverage/`), License (link
   `https://github.com/craabreu/ufedmm/blob/main/LICENSE.md`), and PyPI
   (`https://img.shields.io/pypi/v/ufedmm.svg`, linking to
   `https://pypi.org/project/ufedmm`). Then keep the three conda badges.
2. In the cvpack section (around lines 78–85), replace
   `[CVPack](https://cvpack.readthedocs.io/en/latest/)` with
   `[CVPack](https://github.com/craabreu/cvpack)`.
3. Add an "Installation" section if there isn't one; otherwise update it:
   ```markdown
   ### Installation

   With conda:

       conda install -c conda-forge -c mdtools ufedmm

   With pip:

       pip install ufedmm
   ```
4. Add a final "Citing UFEDMM" section before the link-reference definitions at
   the bottom. It gives the full reference from Global Constraints with a doi
   link, followed by the `bajpai2023` BibTeX entry above in a ```` ```bibtex ````
   block.

In `docs/installation.rst`, add a pip option after the mamba block:
```rst
Or, with pip:

.. code-block:: bash

    pip install ufedmm
```
Replace any remaining `https://craabreu.github.io/ufedmm/` links (without
`latest`) anywhere in the repo:
`git grep -n 'craabreu.github.io/ufedmm/[^l]\|craabreu.github.io/ufedmm/$\|craabreu.github.io/ufedmm)'`.
Point each one at `/latest` or its subpath.

- [ ] **Step 5: Check**

Run: `$T/python -I $SCRATCH/check_meta.py /data3/craabreu/github_repos/ufedmm`
Expected: `metadata OK`.
Run: `$SCRATCH/pkgenv/bin/pip install -q cffconvert && $SCRATCH/pkgenv/bin/cffconvert --validate -i CITATION.cff`
Expected: `Citation metadata are valid according to schema version 1.2.0.`
Run: `rm -rf $SCRATCH/html && $SCRATCH/docsenv/bin/sphinx-build -q docs $SCRATCH/html 2>&1 | grep -i -E 'bajpai|citation' ; grep -c 'Bajpai' $SCRATCH/html/index.html`
Expected: no warnings about the citation, and a count of at least 1.
Run: `rm -rf $SCRATCH/dist && $SCRATCH/pkgenv/bin/python -m build -o $SCRATCH/dist . && $SCRATCH/pkgenv/bin/twine check $SCRATCH/dist/* && $SCRATCH/pkgenv/bin/python -I $SCRATCH/check_wheel.py $SCRATCH/dist`
Expected: `PASSED` twice, then `wheel OK`. The README renders as the PyPI long
description, so twine's check covers it. Afterwards, restore `_version.py` with
`git checkout ufedmm/_version.py` if the build rewrote it.
Run: `cd devtools/conda-recipes/anaconda && UFEDMM_VERSION=v0.4.0 /data3/craabreu/envs/condabuild/bin/conda-build . -c conda-forge -c mdtools --no-anaconda-upload --output-folder $SCRATCH/condaout && cd -`
Expected: a `ufedmm-0.4.0-*.conda` file in `$SCRATCH/condaout/noarch`, and the
recipe's import test passes.
Run: `$T/pytest -p no:cacheprovider -q && $T/ruff check ufedmm`
Expected: all tests pass and lint is clean.

- [ ] **Step 6: Commit**

```bash
git add CHANGELOG.md CITATION.cff README.md docs ufedmm/_version.py
git commit -m "Add changelog and citation metadata and bump version to 0.4.0"   # plus trailer
```

### Task 11: Open PR 3, then do the post-merge deploy (gated)

- [ ] **Step 1: Push and open PR 3**

```bash
git push -u origin release-deploys-docs
gh pr create --base main --title "Deploy on published releases, publish to PyPI and version the docs" --body "<summary of Tasks 8–10; the actions the user must take: PYPI_TOKEN secret, confirm ANACONDA_TOKEN; the planned gh-pages migration and deep-link breakage; PR trailer>"
gh pr checks --watch
```
Expected: all checks green. The Docs job builds but prints
`Deployment is deactivated on pull requests`.

- [ ] **Step 2: STOP and get the user's approval for the outward-facing steps**

Ask the user to confirm all of the following: PRs #38–#41 are merged (with
merge commits, so the `.git-blame-ignore-revs` SHA stays valid), `PYPI_TOKEN` is
set, and `ANACONDA_TOKEN` is valid. Don't continue without an explicit yes.

- [ ] **Step 3: The user publishes the v0.4.0 GitHub Release, and you watch the deploys**

The release triggers Docs, Anaconda and PyPI. Docs deploys `docs/v0.4.0/` and
`docs/latest/`, while the old flat pages and the old `docs/index.html` keep
serving the root URL in the meantime. Before the user publishes, confirm that
`ufedmm/_version.py` on main reads `0.4.0` and that `date-released` in
`CITATION.cff` matches the release date.
Run: `gh run list -L 5` and then `gh run watch <id>` for each of the three runs.
Expected: all three succeed. Don't seed `latest/` from v0.3.2: that tag's
`docs_env.yaml` has no pytest, so the Docs run would fail.

- [ ] **Step 4: Check that `latest/` is live BEFORE touching the root**

```bash
curl -s -o /dev/null -w '%{http_code}\n' https://craabreu.github.io/ufedmm/latest/
```
Expected: `200`. Pages can take a few minutes; use Monitor with an until-loop.
If you don't get `200`, stop and report to the user. Don't migrate.

- [ ] **Step 5: Migrate gh-pages (only after Step 4 returns 200)**

```bash
cd /data3/craabreu/github_repos/ufedmm
git fetch origin gh-pages
git worktree add $SCRATCH/gh-pages origin/gh-pages -b gh-pages-migrate
cd $SCRATCH/gh-pages
git ls-files docs | grep -v -E '^docs/(\.nojekyll|development/|latest/|v[0-9])' | xargs git rm -q
git -C /data3/craabreu/github_repos/openxps show origin/gh-pages:docs/index.html > docs/index.html
test -f docs/.nojekyll || touch docs/.nojekyll
git add docs/index.html docs/.nojekyll
git commit -m "Move to versioned docs layout with a redirect to latest"   # plus trailer
git push origin gh-pages-migrate:gh-pages
cd /data3/craabreu/github_repos/ufedmm && git worktree remove $SCRATCH/gh-pages && git branch -D gh-pages-migrate
```

- [ ] **Step 6: Check the site (Review Focus #5)**

```bash
curl -s https://craabreu.github.io/ufedmm/ | grep -o 'url=latest/'
curl -s -o /dev/null -w '%{http_code}\n' https://craabreu.github.io/ufedmm/latest/
curl -s -o /dev/null -w '%{http_code}\n' https://craabreu.github.io/ufedmm/development/
```
Expected: `url=latest/`, `200` and `200`. Use Monitor with an until-loop and
don't sleep in a loop.
