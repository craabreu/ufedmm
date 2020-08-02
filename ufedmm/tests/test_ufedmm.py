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


def ufed_model():
    model = ufedmm.AlanineDipeptideModel()
    mass = 50*unit.dalton*(unit.nanometer/unit.radians)**2
    Ks = 1000*unit.kilojoules_per_mole/unit.radians**2
    Ts = 1500*unit.kelvin
    limit = 180*unit.degrees
    sigma = 18*unit.degrees
    height = 2.0*unit.kilojoules_per_mole
    frequency = 10
    s_phi = ufedmm.DynamicalVariable('s_phi', -limit, limit, mass, Ts, model.phi, Ks, sigma=sigma)
    s_psi = ufedmm.DynamicalVariable('s_psi', -limit, limit, mass, Ts, model.psi, Ks, sigma=sigma)
    return model, ufedmm.UnifiedFreeEnergyDynamics([s_phi, s_psi], 300*unit.kelvin, height, frequency)


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


def test_simulation():
    model = ufedmm.AlanineDipeptideModel(water='tip3p')
    mass = 50*unit.dalton*(unit.nanometer/unit.radians)**2
    Ks = 1000*unit.kilojoules_per_mole/unit.radians**2
    Ts = 1500*unit.kelvin
    dt = 2*unit.femtoseconds
    gamma = 10/unit.picoseconds
    limit = 180*unit.degrees
    sigma = 18*unit.degrees
    height = 2*unit.kilojoules_per_mole
    s_phi = ufedmm.DynamicalVariable('s_phi', -limit, limit, mass, Ts, model.phi, Ks, sigma=sigma)
    s_psi = ufedmm.DynamicalVariable('s_psi', -limit, limit, mass, Ts, model.psi, Ks, sigma=sigma)
    ufed = ufedmm.UnifiedFreeEnergyDynamics([s_phi, s_psi], 300*unit.kelvin, height=0, frequency=1)
    integrator = ufedmm.GeodesicLangevinIntegrator(300*unit.kelvin, gamma, dt)
    platform = openmm.Platform.getPlatformByName('Reference')
    simulation = ufed.simulation(model.topology, model.system, integrator, platform)
    simulation.context.setPositions(model.positions)
    simulation.context.setVelocitiesToTemperature(300*unit.kelvin, 1234)
    simulation.context.getIntegrator().setRandomNumberSeed(1234)
    # print(simulation.context.getState(getVelocities=True).getVelocities(asNumpy=True))
    simulation.step(1)
    energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    assert energy/energy.unit == pytest.approx(-11691.10734)
