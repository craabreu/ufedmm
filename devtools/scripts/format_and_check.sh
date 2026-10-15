#!/usr/bin/env bash
set -e -v
ruff format ufedmm
ruff check --fix ufedmm
