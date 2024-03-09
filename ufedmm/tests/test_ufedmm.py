"""
Unit and regression test for the ufedmm package.
"""

# Import package, test suite, and other packages as needed
import io
import sys
from copy import deepcopy

import numpy as np
import openmm
import pytest
from openmm import unit

import ufedmm


def ufed_model(
    temp=300 * unit.kelvin,
    mass=50 * unit.dalton * (unit.nanometer / unit.radians) ** 2,
    Ks=1000 * unit.kilojoules_per_mole / unit.radians**2,
    Ts=1500 * unit.kelvin,
    limit=180 * unit.degrees,
    sigma=18 * unit.degrees,
    height=0.0,
    frequency=10,
    bias_factor=None,
    enforce_gridless=False,
    constraints=openmm.app.HBonds,
):
    model = ufedmm.AlanineDipeptideModel(constraints=constraints)
    s_phi = ufedmm.DynamicalVariable(
        "s_phi", -limit, limit, mass, Ts, model.phi, Ks, sigma=sigma
    )
    s_psi = ufedmm.DynamicalVariable(
        "s_psi", -limit, limit, mass, Ts, model.psi, Ks, sigma=sigma
    )
    return model, ufedmm.UnifiedFreeEnergyDynamics(
        [s_phi, s_psi],
        temp,
        height,
        frequency,
        bias_factor,
        enforce_gridless=enforce_gridless,
    )


def test_ufedmm_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "ufedmm" in sys.modules


def test_serialization():
    _, old = ufed_model()
    pipe = io.StringIO()
    ufedmm.serialize(old, pipe)
    pipe.seek(0)
    new = ufedmm.deserialize(pipe)
    assert new.__repr__() == old.__repr__()
    for var1, var2 in zip(old.variables, new.variables):
        assert var1.__repr__() == var2.__repr__()


def test_variables():
    model, ufed = ufed_model(constraints=None)
    integrator = ufedmm.MiddleMassiveNHCIntegrator(
        300 * unit.kelvin, 10 * unit.femtoseconds, 1 * unit.femtoseconds
    )
    platform = openmm.Platform.getPlatformByName("Reference")
    simulation = ufed.simulation(model.topology, model.system, integrator, platform)
    simulation.context.setPositions(model.positions)
    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, 11234)
    simulation.step(20)
    cvs = [
        value
        for force in simulation.context.driving_forces
        for value in force.getCollectiveVariableValues(simulation.context)
    ]
    xvars = simulation.context.getState(getPositions=True).getDynamicalVariables()
    assert cvs[0] == pytest.approx(xvars[0])
    assert cvs[2] == pytest.approx(xvars[1])
    state = simulation.context.getState(getPositions=True)
    print(type(state.getPositions(asNumpy=False)))
    simulation.context.setPositions(*state.getPositions(extended=True))
    xvars = simulation.context.getState(getPositions=True).getDynamicalVariables()
    assert cvs[0] == pytest.approx(xvars[0])
    assert cvs[2] == pytest.approx(xvars[1])


def test_NHC_integrator():
    model, ufed = ufed_model(constraints=None)
    nonbonded = next(
        filter(lambda f: isinstance(f, openmm.NonbondedForce), model.system.getForces())
    )
    nonbonded.setForceGroup(1)
    integrator = ufedmm.MiddleMassiveNHCIntegrator(
        300 * unit.kelvin,
        10 * unit.femtoseconds,
        1 * unit.femtoseconds,
        respa_loops=[5, 1],
    )
    platform = openmm.Platform.getPlatformByName("Reference")
    simulation = ufed.simulation(model.topology, model.system, integrator, platform)
    simulation.context.setPositions(model.positions)
    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, 1234)
    simulation.step(100)
    energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    assert energy / energy.unit == pytest.approx(7.690, abs=0.2)


def test_GGMT_integrator():
    model, ufed = ufed_model(constraints=None)
    nonbonded = next(
        filter(lambda f: isinstance(f, openmm.NonbondedForce), model.system.getForces())
    )
    nonbonded.setForceGroup(1)
    integrator = ufedmm.MiddleMassiveGGMTIntegrator(
        300 * unit.kelvin,
        10 * unit.femtoseconds,
        1 * unit.femtoseconds,
        respa_loops=[5, 1],
    )
    platform = openmm.Platform.getPlatformByName("Reference")
    simulation = ufed.simulation(model.topology, model.system, integrator, platform)
    simulation.context.setPositions(model.positions)
    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, 1234)
    simulation.step(100)
    energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    assert energy / energy.unit == pytest.approx(-20.70, abs=0.2)


def test_velocities():
    model, ufed = ufed_model(height=2.0 * unit.kilocalorie_per_mole, constraints=None)
    integrator = ufedmm.MiddleMassiveNHCIntegrator(
        300 * unit.kelvin, 10 * unit.femtoseconds, 1 * unit.femtoseconds
    )
    platform = openmm.Platform.getPlatformByName("Reference")
    simulation = ufed.simulation(model.topology, model.system, integrator, platform)
    simulation.context.setPositions(model.positions)
    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, 1234)
    state = simulation.context.getState(getVelocities=True)

    velocities = state.getVelocities(asNumpy=True, extended=True).value_in_unit(
        unit.nanometer / unit.picosecond
    )
    masses = np.array(
        [
            model.system.getParticleMass(i).value_in_unit(unit.dalton)
            for i in range(model.system.getNumParticles())
        ]
    )

    n = len(velocities) - 2
    com_velocity = (masses[:n, None] * velocities[:n, :]).sum(axis=0) / masses[:n].sum()
    assert com_velocity[0] == pytest.approx(0, abs=1e-6)

    kB = unit.MOLAR_GAS_CONSTANT_R.value_in_unit_system(unit.md_unit_system)
    two_ke = (masses[:n] * np.linalg.norm(velocities[:n, :], axis=1) ** 2).sum()
    temperature = two_ke / (3 * n * kB)
    assert temperature == pytest.approx(300, abs=1e-6)

    phi_v, psi_v = velocities[-2:, 0]
    assert masses[n] * phi_v**2 / kB == pytest.approx(1500, abs=1e-6)
    assert masses[n] * psi_v**2 / kB == pytest.approx(1500, abs=1e-6)


def test_gridded_metadynamics():
    model, ufed = ufed_model(height=2.0 * unit.kilocalorie_per_mole, constraints=None)
    integrator = ufedmm.MiddleMassiveNHCIntegrator(
        300 * unit.kelvin, 10 * unit.femtoseconds, 1 * unit.femtoseconds
    )
    platform = openmm.Platform.getPlatformByName("Reference")
    simulation = ufed.simulation(model.topology, model.system, integrator, platform)
    simulation.context.setPositions(model.positions)
    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, 1234)
    simulation.step(100)
    energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    assert energy / energy.unit == pytest.approx(717.43, abs=0.1)


# def test_gridless_metadynamics():
#     model, ufed = ufed_model(
#         height=2.0*unit.kilocalorie_per_mole, enforce_gridless=True
#     )
#     integrator = ufedmm.MiddleMassiveNHCIntegrator(
#         300*unit.kelvin, 10*unit.femtoseconds, 1*unit.femtoseconds
#     )
#     platform = openmm.Platform.getPlatformByName('Reference')
#     simulation = ufed.simulation(model.topology, model.system, integrator, platform)
#     simulation.context.setPositions(model.positions)
#     simulation.context.setVelocitiesToTemperature(300*unit.kelvin, 1234)
#     simulation.step(100)
#     energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
#     assert energy/energy.unit == pytest.approx(59.74040)


def test_well_tempered_metadynamics():
    model, ufed = ufed_model(
        height=2.0 * unit.kilocalorie_per_mole, bias_factor=10, constraints=None
    )
    integrator = ufedmm.MiddleMassiveNHCIntegrator(
        300 * unit.kelvin, 10 * unit.femtoseconds, 1 * unit.femtoseconds
    )
    platform = openmm.Platform.getPlatformByName("Reference")
    simulation = ufed.simulation(model.topology, model.system, integrator, platform)
    simulation.context.setPositions(model.positions)
    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, 1234)
    simulation.step(100)
    energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    assert energy / energy.unit == pytest.approx(172.7, abs=0.1)


def test_rmsd_forces():
    model, ufed = ufed_model(constraints=None)
    rmsd = openmm.RMSDForce(  # Level 1
        model.positions, np.arange(model.system.getNumParticles())
    )
    cvforce = openmm.CustomCVForce("rmsd + cvforce")
    cvforce.addCollectiveVariable("rmsd", deepcopy(rmsd))  # Level 2
    inner_cvforce = openmm.CustomCVForce("rmsd")
    inner_cvforce.addCollectiveVariable("rmsd", deepcopy(rmsd))  # Level 3
    cvforce.addCollectiveVariable("cvforce", inner_cvforce)
    model.system.addForce(rmsd)
    model.system.addForce(cvforce)
    integrator = ufedmm.MiddleMassiveNHCIntegrator(
        300 * unit.kelvin, 10 * unit.femtoseconds, 1 * unit.femtoseconds
    )
    platform = openmm.Platform.getPlatformByName("Reference")
    simulation = ufed.simulation(model.topology, model.system, integrator, platform)
    simulation.context.setPositions(model.positions)
    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, 1234)
    simulation.step(1)
