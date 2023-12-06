import numpy as np
import openmm
import pytest

import ufedmm
from ufedmm.ufedmm import _standardized


def S(z):
    if z < 0:
        return 1
    elif z < 1:
        return 1 - 10 * z**3 + 15 * z**4 - 6 * z**5
    else:
        return 0


def perform_test(sigma, epsilon, charge0, charge1, rs, rc):
    nonbonded = openmm.NonbondedForce()
    nonbonded.addParticle(charge0, sigma, epsilon)
    nonbonded.addParticle(charge1, sigma, epsilon)
    nonbonded.setNonbondedMethod(nonbonded.CutoffNonPeriodic)
    platform = openmm.Platform.getPlatformByName("Reference")
    system = openmm.System()
    system.addParticle(1)
    system.addParticle(1)
    system.addForce(nonbonded)
    ufedmm.add_inner_nonbonded_force(system, rs, rc, 1)
    context = openmm.Context(system, openmm.CustomIntegrator(0), platform)
    ONE_4PI_EPS0 = 138.93545764438198
    for r in np.linspace(sigma, rc, 101):
        context.setPositions([[0, 0, 0], [r, 0, 0]])
        state = context.getState(getForces=True, groups={1})
        force = state.getForces()[1].x
        F = (
            24 * epsilon * (2 * (sigma / r) ** 12 - (sigma / r) ** 6) / r
            + ONE_4PI_EPS0 * charge0 * charge1 / r**2
        )
        assert force == pytest.approx(F * S((r - rs) / (rc - rs)))


def test_inner_lennard_jones():
    perform_test(1.0, 1.0, 0, 0, 2.0, 2.5)


def test_inner_coulomb():
    perform_test(1.0, 0.0, -1.0, 1.0, 2.0, 2.5)


def test_inner_exceptions():
    model = ufedmm.AlanineDipeptideModel()
    nbforce = next(
        filter(lambda f: isinstance(f, openmm.NonbondedForce), model.system.getForces())
    )
    rs = 0.2
    rc = 0.4
    ufedmm.add_inner_nonbonded_force(model.system, rs, rc, 1)
    model.system.getForce(model.system.getNumForces() - 1).setForceGroup(3)
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(model.system, openmm.CustomIntegrator(0), platform)
    context.setPositions(model.positions)
    forces1 = _standardized(context.getState(getForces=True, groups={3}).getForces())
    forces2 = [0 * f for f in forces1]
    ONE_4PI_EPS0 = 138.93545764438198
    for index in range(nbforce.getNumExceptions()):
        i, j, chargeprod, sigma, epsilon = map(
            _standardized, nbforce.getExceptionParameters(index)
        )
        rij = _standardized(model.positions[i] - model.positions[j])
        r = np.linalg.norm(rij)
        z = (r - rs) / (rc - rs)
        F = (
            S(z)
            * (
                24 * epsilon * (2 * (sigma / r) ** 12 - (sigma / r) ** 6) / r
                + ONE_4PI_EPS0 * chargeprod / r**2
            )
            * rij
            / r
        )
        forces2[i] += F
        forces2[j] -= F
    for f1, f2 in zip(forces1, forces2):
        for i in range(3):
            assert f1[i] == pytest.approx(f2[i])
