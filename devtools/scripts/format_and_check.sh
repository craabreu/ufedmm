#!/usr/bin/env bash
set -e -v
for tool in isort black flake8; do
    $tool ufedmm
done
