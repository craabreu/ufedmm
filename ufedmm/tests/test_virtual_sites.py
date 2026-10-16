import re

import numpy as np
import openmm
import pytest
from openmm import app, unit

import ufedmm

from .common import GPU_PLATFORMS

TEMP = 300 * unit.kelvin
DT = 1.5 * unit.femtoseconds
TAU = 160 * unit.femtoseconds
GAMMA = 10 / unit.picoseconds


@pytest.fixture(scope="module")
def flexible_tip4p_box():
    # TIP4P-type water has a massless virtual site carrying the negative charge
    forcefield = app.ForceField("amber14/tip4pfb.xml")
    modeller = app.Modeller(app.Topology(), [])
    modeller.addSolvent(
        forcefield, model="tip4pew", boxSize=openmm.Vec3(3, 3, 3) * unit.nanometer
    )
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=0.9 * unit.nanometer,
        constraints=None,
        rigidWater=False,
        removeCMMotion=False,
    )
    integrator = openmm.VerletIntegrator(1 * unit.femtoseconds)
    context = openmm.Context(
        system, integrator, openmm.Platform.getPlatformByName("CPU")
    )
    context.setPositions(modeller.positions)
    openmm.LocalEnergyMinimizer.minimize(context, maxIterations=200)
    positions = context.getState(getPositions=True).getPositions()
    return system, positions


INTEGRATORS = pytest.mark.parametrize(
    "make_integrator",
    [
        lambda: ufedmm.MiddleMassiveNHCIntegrator(TEMP, TAU, DT),
        lambda: ufedmm.MiddleMassiveGGMTIntegrator(TEMP, TAU, DT),
        lambda: ufedmm.RegulatedNHLIntegrator(TEMP, TAU, GAMMA, DT, 1),
        lambda: ufedmm.GeodesicLangevinIntegrator(TEMP, GAMMA, DT, num_rattles=0),
        lambda: ufedmm.MiddleMassiveNHCIntegrator(
            TEMP, TAU, DT, respa_loops=[2, 1], unroll_loops=False
        ),
    ],
    ids=["NHC", "GGMT", "RegulatedNHL", "GeodesicLangevin-no-rattle", "NHC-RESPA"],
)


@INTEGRATORS
def test_positions_are_constrained_before_forces_are_used(make_integrator):
    # GPU platforms only update virtual sites in "constrain positions" steps, so
    # every position update must be followed by one before forces are used again.
    integrator = make_integrator()
    pending = False
    for i in range(integrator.getNumComputations()):
        kind, variable, expression = integrator.getComputationStep(i)
        if kind == openmm.CustomIntegrator.ConstrainPositions:
            pending = False
        elif kind == openmm.CustomIntegrator.ComputePerDof and variable == "x":
            pending = True
        elif re.search(r"\bf\d*\b", expression):
            assert not pending, f"step {i} uses forces before positions are constrained"
    assert not pending, "the step ends without constraining the last position update"


@pytest.mark.skipif(not GPU_PLATFORMS, reason="requires a CUDA or OpenCL platform")
@pytest.mark.parametrize("platform_name", GPU_PLATFORMS)
@INTEGRATORS
def test_integrators_keep_virtual_sites_up_to_date(
    flexible_tip4p_box, platform_name, make_integrator
):
    system, positions = flexible_tip4p_box
    integrator = make_integrator()
    context = openmm.Context(
        system, integrator, openmm.Platform.getPlatformByName(platform_name)
    )
    context.setPositions(positions)
    context.setVelocitiesToTemperature(TEMP, 1234)
    num_waters = system.getNumParticles() // 4
    for _ in range(12):
        integrator.step(50)
        energy = context.getState(getEnergy=True).getPotentialEnergy()
        # A Coulomb collapse onto stale virtual sites drives this to -1e5 and below
        assert np.isfinite(energy._value)
        assert energy / num_waters > -100 * unit.kilojoules_per_mole
