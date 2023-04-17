#!/usr/bin/env bash
set -e -v
isort ./ufedmm
black --line-length 100 ./ufedmm
flake8 --ignore=E203,W503 ./ufedmm
# pylint --rcfile=devtools/linters/pylintrc ./ufedmm
