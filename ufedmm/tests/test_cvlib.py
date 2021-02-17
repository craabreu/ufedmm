import ufedmm
import pytest

from copy import deepcopy
from simtk import openmm
from ufedmm import cvlib


def potential_energy(system, positions, force_cls):
    syscopy = deepcopy(system)
    for force in syscopy.getForces():
        if isinstance(force, force_cls):
            force.setForceGroup(31)
    integrator = openmm.CustomIntegrator(0)
    platform = openmm.Platform.getPlatformByName('Reference')
    context = openmm.Context(syscopy, integrator, platform)
    context.setPositions(positions)
    return context.getState(getEnergy=True, groups={31}).getPotentialEnergy()


def test_in_out_lennard_jones_force():
    model = ufedmm.AlanineDipeptideModel(water='tip3p')
    before = potential_energy(model.system, model.positions, openmm.NonbondedForce)
    solute_atoms = [atom.index for atom in model.topology.atoms() if atom.residue.name != 'HOH']
    nbforce = next(filter(lambda f: isinstance(f, openmm.NonbondedForce), model.system.getForces()))
    in_out_LJ = cvlib.InOutLennardJonesForce(solute_atoms, nbforce)
    model.system.addForce(in_out_LJ)
    after = potential_energy(model.system, model.positions, (openmm.NonbondedForce, openmm.CustomNonbondedForce))
    assert after/after.unit == pytest.approx(before/before.unit, 1E-2)
