import numpy as np
import pytest

from ufedmm import integrators
from simtk import openmm


def perform_test(sigma, epsilon, charge0, charge1, rs, rc):
    nonbonded = openmm.NonbondedForce()
    nonbonded.addParticle(charge0, sigma, epsilon)
    nonbonded.addParticle(charge1, sigma, epsilon)
    nonbonded.setNonbondedMethod(nonbonded.CutoffNonPeriodic)
    platform = openmm.Platform.getPlatformByName('Reference')
    system = openmm.System()
    system.addParticle(1)
    system.addParticle(1)
    system.addForce(nonbonded)
    integrators.add_inner_nonbonded_force(system, rs, rc, 1)
    context = openmm.Context(system, openmm.CustomIntegrator(0), platform)
    ONE_4PI_EPS0 = 138.93545764438198
    for r in np.linspace(sigma, rc, 101):
        context.setPositions([[0, 0, 0], [r, 0, 0]])
        state = context.getState(getForces=True, groups={1})
        force = state.getForces()[1].x
        z = (r - rs)/(rc - rs)
        S = 1 - 10*z**3 + 15*z**4 - 6*z**5 if z > 0 else 1
        F = 24*epsilon*(2*(sigma/r)**12 - (sigma/r)**6)/r + ONE_4PI_EPS0*charge0*charge1/r**2
        assert force == pytest.approx(F*S)


def test_inner_lennard_jones():
    perform_test(1.0, 1.0, 0, 0, 2.0, 2.5)


def test_inner_coulomb():
    perform_test(1.0, 0.0, -1.0, 1.0, 2.0, 2.5)
