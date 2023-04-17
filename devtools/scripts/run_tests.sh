#!/usr/bin/env bash

pytest -v -s --cov=ufedmm --cov-report=term-missing --cov-report=html --pyargs --doctest-modules "$@" ufedmm
