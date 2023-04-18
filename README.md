Unified Free Energy Dynamics with OpenMM
========================================

[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/craabreu/ufedmm/workflows/Linux/badge.svg)](https://github.com/craabreu/ufedmm/actions?query=workflow%3ALinux)
[![GitHub Actions Build Status](https://github.com/craabreu/ufedmm/workflows/MacOS/badge.svg)](https://github.com/craabreu/ufedmm/actions?query=workflow%3AMacOS)
[![GitHub Actions Build Status](https://github.com/craabreu/ufedmm/workflows/Windows/badge.svg)](https://github.com/craabreu/ufedmm/actions?query=workflow%3AWindows)
[![GitHub Actions Build Status](https://github.com/craabreu/ufedmm/workflows/Docs/badge.svg)](https://craabreu.github.io/ufedmm/)
[![Coverage](https://craabreu.github.io/ufedmm/coverage/coverage.svg)](https://craabreu.github.io/ufedmm/coverage/)
[![License](https://img.shields.io/badge/License-MIT-yellowgreen.svg?style=flat)](https://github.com/craabreu/ufedmm/blob/main/LICENSE.md)

UFEDMM extends [OpenMM's Python API] so that the user can easily run
efficient simulations in extended phase spaces, perform enhanced sampling
of systems with barriers and rare events, and compute accurate free-energy
surfaces for collective variables or reaction coordinates.

#### Extended Phase-Space Dynamics

The concept of extended phase space is a powerful tool in Molecular
Dynamics. It consists in treating arbitrary variables as coordinates of
fictitious particles, assigning masses to these particles, and solving
equations of motion which encode their interactions with the system
molecules. Differently from _collective variables_, which are functions
of atomic coordinates, these _extra dynamical variables_ are independent
ones. Together with their conjugate momenta, they add new dimensions to
the system's phase space.

#### Free Energy Calculations

Free energy is an important thermodynamic property that quantifies the
relative likelihood of different states of a system. UFEDMM uses
extended phase-space dynamics to facilitate the calculation of free
energy as a function of extra dynamical variables. Under certain
assumptions, this is a suitable approximation for the free energy as
a function of collective variables, also known as potential of mean
force (PMF).

#### Enhanced Sampling of Rare Events

UFEDMM combines two methods to efficiently overcome free-energy barriers.
The [TAMD]/[d-AFED] method heats the extended variables to a higher
temperature than the one specified for the molecules. The [Metadynamics]
method floods free-energy basins with potential energy so that barriers
are eventually smoothed out. This is the Unified Free Energy Dynamics
([UFED]) method: heating and flooding, all at once.

## Methods and Algorithms

Interaction between a fictitious particle and the actual molecules is
enacted by adding, to the total potential energy of the system, a new
term that depends both on the corresponding dynamical variable and at
least one collective variable. With OpenMM's [CustomCVForce] class,
adding such a term is as simple as writing down a mathematical expression.
All the low-level coding and compilation takes place automatically in the
background.

UFEDMM build on the customization capability of OpenMM to enable efficient
[UFED] simulations in GPU's and other parallel computation platforms. It
is efficient because it makes OpenMM treat extra dynamical variables like
normal atomic coordinates, thus avoiding the computational overhead of
dealing with [Context] parameters.

[TAMD]/[d-AFED] is optionally enabled by assigning distinct temperatures
to the molecules and to the extra dynamical variables. For this, UFEDMM
provides special [CustomIntegrator] subclasses, given that the intrinsic
OpenMM integrators cannot handle multiple temperatures. Extended-space
[Metadynamics] is enabled by explicitly defining the height and widths of
Gaussian hills to be deposited over time, as well as the deposition period.

For the post-processing of [UFED] simulations, a free energy analysis tool
is provided, which is based on mean-force estimation and radial basis set
reconstruction of free energy (hyper)surfaces.

## Collective Variable Package (cvpack)

In OpenMM, a collective variable (CV) involved in a [CustomCVForce] are
nothing but objects of some `Force` (or, particularly, `CustomForce`)
subclass. The user is free to define any CV for a [UFED] simulation. For
convenience, through, UFEDMM provides a module with several predefined
CV's, such as:

* Square radius of gyration of a group of atoms
* Number of contacts between two groups of atoms
* Different flavors of alpha-helix content measures, based on angles,
dihedrals, and hydrogen bonds

The `ufedmm.cvpack` module can be viewed as a standalone package of
general applicability, not restricted to `ufedmm` simulations.

## Documentation

https://craabreu.github.io/ufedmm/

#### Copyright

This is an open-source (MIT licensed) project. Contribution is welcome.

#### Acknowledgements

Project structure based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.0.

[OpenMM's Python API]: http://docs.openmm.org/latest/api-python/index.html
[TAMD]: http://doi.org/10.1016/j.cplett.2006.05.062
[d-AFED]: http://doi.org/10.1021/jp805039u
[Metadynamics]: http://doi.org/10.1021/jp045424k
[UFED]: http://doi.org/10.1063/1.4733389
[Context]: http://docs.openmm.org/latest/api-python/generated/openmm.openmm.Context.html
[CustomCVForce]: http://docs.openmm.org/latest/api-python/generated/openmm.openmm.CustomCVForce.html
[CustomIntegrator]: http://docs.openmm.org/latest/api-python/generated/openmm.openmm.CustomIntegrator.html
