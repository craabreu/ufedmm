import ufedmm
import pytest

from copy import deepcopy
from simtk import openmm
from ufedmm import cvlib

from simtk.openmm import app


def test_radius_of_gyration():
    model = ufedmm.AlanineDipeptideModel()

    R = model.positions._value
    N = len(R)
    Rmean = sum(R, openmm.Vec3(0, 0, 0))/N
    RgSqVal = 0.0
    for r in R:
        dr = r - Rmean
        RgSqVal += dr.x**2 + dr.y**2 + dr.z**2
    RgSqVal /= N

    RgSq = cvlib.SquareRadiusOfGyration(range(model.topology._numAtoms))
    RgSq.setForceGroup(1)
    model.system.addForce(RgSq)
    Rg = cvlib.RadiusOfGyration(range(model.topology._numAtoms))
    Rg.setForceGroup(2)
    model.system.addForce(Rg)
    integrator = openmm.CustomIntegrator(0)
    platform = openmm.Platform.getPlatformByName('Reference')
    context = openmm.Context(model.system, integrator, platform)
    context.setPositions(model.positions)
    RgSq = context.getState(getEnergy=True, groups={1}).getPotentialEnergy()._value
    assert RgSq == pytest.approx(RgSqVal)
    Rg = context.getState(getEnergy=True, groups={2}).getPotentialEnergy()._value
    assert Rg*Rg == pytest.approx(RgSqVal)


def potential_energy(system, positions, force_cls, scaling=None):
    syscopy = deepcopy(system)
    for force in syscopy.getForces():
        if isinstance(force, force_cls):
            force.setForceGroup(31)
    integrator = openmm.CustomIntegrator(0)
    platform = openmm.Platform.getPlatformByName('Reference')
    context = openmm.Context(syscopy, integrator, platform)
    context.setPositions(positions)
    if scaling is not None:
        context.setParameter('inOutCoulombScaling', scaling)
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


def test_in_out_coulomb_force():
    model = ufedmm.AlanineDipeptideModel(water='tip3p')
    before = potential_energy(model.system, model.positions, openmm.NonbondedForce)
    solute_atoms = [atom.index for atom in model.topology.atoms() if atom.residue.name != 'HOH']
    nbforce = next(filter(lambda f: isinstance(f, openmm.NonbondedForce), model.system.getForces()))
    in_out_coul = cvlib.InOutCoulombForce(solute_atoms, nbforce)
    model.system.addForce(in_out_coul)
    after = potential_energy(model.system, model.positions, (openmm.NonbondedForce, openmm.CustomNonbondedForce))
    assert after/after.unit == pytest.approx(before/before.unit, 0.1)
    scaled = potential_energy(model.system, model.positions, openmm.NonbondedForce, scaling=1.0)
    assert scaled/scaled.unit == pytest.approx(before/before.unit, 1E-5)
