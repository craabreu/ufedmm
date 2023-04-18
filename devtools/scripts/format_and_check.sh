#!/usr/bin/env bash
set -e -v
isort --line-length=100 ./ufedmm
black --line-length 100 ./ufedmm
flake8 --max-line-length=100 --ignore=E203,W503 ./ufedmm
# pylint --rcfile=devtools/linters/pylintrc ./ufedmm
