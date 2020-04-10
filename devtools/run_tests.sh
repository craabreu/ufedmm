#!/usr/bin/env bash

# exit when any command fails
set -e

flake8 ufedmm/
isort ufedmm/ufedmm.py
sphinx-build docs/ docs/_build
pytest -v --cov=ufedmm --doctest-modules ufedmm/
