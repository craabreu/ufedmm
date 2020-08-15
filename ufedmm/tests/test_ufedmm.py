"""
Unit and regression test for the ufedmm package.
"""

# Import package, test suite, and other packages as needed
import ufedmm
import pytest
import sys
import io

from simtk import openmm
from simtk import unit


def ufed_model(
    temp=300*unit.kelvin,
    mass=50*unit.dalton*(unit.nanometer/unit.radians)**2,
    Ks=1000*unit.kilojoules_per_mole/unit.radians**2,
    Ts=1500*unit.kelvin,
    limit=180*unit.degrees,
    sigma=18*unit.degrees,
    height=0.0,
    frequency=10,
    enforce_gridless=False,
):
    model = ufedmm.AlanineDipeptideModel()
    s_phi = ufedmm.DynamicalVariable('s_phi', -limit, limit, mass, Ts, model.phi, Ks, sigma=sigma)
    s_psi = ufedmm.DynamicalVariable('s_psi', -limit, limit, mass, Ts, model.psi, Ks, sigma=sigma)
    return model, ufedmm.UnifiedFreeEnergyDynamics([s_phi, s_psi], temp, height, frequency,
                                                   enforce_gridless=enforce_gridless)


def test_ufedmm_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "ufedmm" in sys.modules


def test_serialization():
    model, old = ufed_model()
    pipe = io.StringIO()
    ufedmm.serialize(old, pipe)
    pipe.seek(0)
    new = ufedmm.deserialize(pipe)
    assert new.__repr__() == old.__repr__()
    for var1, var2 in zip(old.variables, new.variables):
        assert var1.__repr__() == var2.__repr__()


def test_variables():
    model, ufed = ufed_model()
    integrator = ufedmm.MiddleMassiveNHCIntegrator(300*unit.kelvin, 10*unit.femtoseconds, 1*unit.femtoseconds)
    platform = openmm.Platform.getPlatformByName('Reference')
    simulation = ufed.simulation(model.topology, model.system, integrator, platform)
    simulation.context.setPositions(model.positions)
    simulation.context.setVelocitiesToTemperature(300*unit.kelvin, 11234)
    simulation.step(20)
    cvs = simulation.context.driving_force.getCollectiveVariableValues(simulation.context)
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
    model, ufed = ufed_model()
    integrator = ufedmm.MiddleMassiveNHCIntegrator(300*unit.kelvin, 10*unit.femtoseconds, 1*unit.femtoseconds)
    platform = openmm.Platform.getPlatformByName('Reference')
    simulation = ufed.simulation(model.topology, model.system, integrator, platform)
    simulation.context.setPositions(model.positions)
    simulation.context.setVelocitiesToTemperature(300*unit.kelvin, 1234)
    simulation.step(100)
    energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    assert energy/energy.unit == pytest.approx(-23.43190)


def test_GGMT_integrator():
    model, ufed = ufed_model()
    integrator = ufedmm.MiddleMassiveGGMTIntegrator(300*unit.kelvin, 10*unit.femtoseconds, 1*unit.femtoseconds)
    platform = openmm.Platform.getPlatformByName('Reference')
    simulation = ufed.simulation(model.topology, model.system, integrator, platform)
    simulation.context.setPositions(model.positions)
    simulation.context.setVelocitiesToTemperature(300*unit.kelvin, 1234)
    simulation.step(100)
    energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    assert energy/energy.unit == pytest.approx(-22.52247)


def test_gridded_metadynamics():
    model, ufed = ufed_model(height=2.0*unit.kilocalorie_per_mole)
    integrator = ufedmm.MiddleMassiveNHCIntegrator(300*unit.kelvin, 10*unit.femtoseconds, 1*unit.femtoseconds)
    platform = openmm.Platform.getPlatformByName('Reference')
    simulation = ufed.simulation(model.topology, model.system, integrator, platform)
    simulation.context.setPositions(model.positions)
    simulation.context.setVelocitiesToTemperature(300*unit.kelvin, 1234)
    simulation.step(100)
    energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    assert energy/energy.unit == pytest.approx(672.48299)


# def test_gridless_metadynamics():
#     model, ufed = ufed_model(height=2.0*unit.kilocalorie_per_mole, enforce_gridless=True)
#     integrator = ufedmm.MiddleMassiveNHCIntegrator(300*unit.kelvin, 10*unit.femtoseconds, 1*unit.femtoseconds)
#     platform = openmm.Platform.getPlatformByName('Reference')
#     simulation = ufed.simulation(model.topology, model.system, integrator, platform)
#     simulation.context.setPositions(model.positions)
#     simulation.context.setVelocitiesToTemperature(300*unit.kelvin, 1234)
#     simulation.step(100)
#     energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
#     assert energy/energy.unit == pytest.approx(59.74040)
