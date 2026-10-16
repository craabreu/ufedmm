import numpy as np
import openmm
import pytest
from openmm import unit

import ufedmm

TEMP = 300 * unit.kelvin
DT = 1 * unit.femtoseconds
TAU = 100 * unit.femtoseconds
GAMMA = 10 / unit.picoseconds


def ufed_for(model):
    mass = 50 * unit.dalton * (unit.nanometer / unit.radians) ** 2
    Ks = 1000 * unit.kilojoules_per_mole / unit.radians**2
    s_phi = ufedmm.DynamicalVariable(
        "s_phi", -np.pi, np.pi, mass, 1500 * unit.kelvin, model.phi, Ks
    )
    return ufedmm.UnifiedFreeEnergyDynamics([s_phi], TEMP)


@pytest.mark.parametrize(
    "make_integrator",
    [
        lambda: ufedmm.MiddleMassiveNHCIntegrator(TEMP, TAU, DT),
        lambda: ufedmm.MiddleMassiveGGMTIntegrator(TEMP, TAU, DT),
        lambda: ufedmm.RegulatedNHLIntegrator(TEMP, TAU, GAMMA, DT, 2),
        lambda: ufedmm.GeodesicLangevinIntegrator(TEMP, GAMMA, DT, num_rattles=0),
    ],
    ids=["NHC", "GGMT", "RegulatedNHL", "GeodesicLangevin-no-rattle"],
)
def test_constrained_system_is_rejected_without_rattle(make_integrator):
    model = ufedmm.AlanineDipeptideModel(constraints=openmm.app.HBonds)
    ufed = ufed_for(model)
    with pytest.raises(ValueError, match="constraints"):
        ufed.simulation(model.topology, model.system, make_integrator())


def test_constrained_system_is_accepted_with_rattle():
    model = ufedmm.AlanineDipeptideModel(constraints=openmm.app.HBonds)
    ufed = ufed_for(model)
    integrator = ufedmm.GeodesicLangevinIntegrator(TEMP, GAMMA, DT, num_rattles=1)
    ufed.simulation(model.topology, model.system, integrator)


def test_unconstrained_system_is_accepted_without_rattle():
    model = ufedmm.AlanineDipeptideModel(constraints=None)
    ufed = ufed_for(model)
    integrator = ufedmm.MiddleMassiveNHCIntegrator(TEMP, TAU, DT)
    ufed.simulation(model.topology, model.system, integrator)
