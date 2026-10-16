"""Helpers shared by the test modules."""

import openmm
import pytest
from openmm import unit

import ufedmm

KB = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(unit.kilojoules_per_mole / unit.kelvin)
METAD_GROUP = 31  # force group assigned to metadynamics biases in the tests

TEMP = 300 * unit.kelvin
TAU = 10 * unit.femtoseconds
DT = 1 * unit.femtoseconds


def ufed_model(  # noqa: PLR0913, PLR0917
    temp=TEMP,
    mass=50 * unit.dalton * (unit.nanometer / unit.radians) ** 2,
    Ks=1000 * unit.kilojoules_per_mole / unit.radians**2,
    Ts=1500 * unit.kelvin,
    limit=180 * unit.degrees,
    sigma=18 * unit.degrees,
    height=0.0,
    frequency=10,
    bias_factor=None,
    enforce_gridless=False,
    constraints=None,
    bias_psi=True,
):
    model = ufedmm.AlanineDipeptideModel(constraints=constraints)
    s_phi = ufedmm.DynamicalVariable(
        "s_phi", -limit, limit, mass, Ts, model.phi, Ks, sigma=sigma
    )
    s_psi = ufedmm.DynamicalVariable(
        "s_psi",
        -limit,
        limit,
        mass,
        Ts,
        model.psi,
        Ks,
        sigma=sigma if bias_psi else None,
    )
    return model, ufedmm.UnifiedFreeEnergyDynamics(
        [s_phi, s_psi],
        temp,
        height,
        frequency,
        bias_factor,
        enforce_gridless=enforce_gridless,
    )


def reference_simulation(model, ufed, integrator):
    platform = openmm.Platform.getPlatformByName("Reference")
    simulation = ufed.simulation(model.topology, model.system, integrator, platform)
    simulation.context.setPositions(model.positions)
    simulation.context.setVelocitiesToTemperature(TEMP, 1234)
    return simulation


def nonbonded_force(system):
    return next(f for f in system.getForces() if isinstance(f, openmm.NonbondedForce))


def phase_space_point(simulation):
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    positions = state.getPositions(asNumpy=True, extended=True, split=False)
    velocities = state.getVelocities(asNumpy=True, extended=True)
    return (
        positions.value_in_unit(unit.nanometers),
        velocities.value_in_unit(unit.nanometers / unit.picosecond),
    )


def torsion(model, name, atom_names):
    atoms = [(a.name, a.residue.name) for a in model.topology.atoms()]
    force = openmm.CustomTorsionForce("theta")
    force.addTorsion(*[atoms.index(a) for a in atom_names], [])
    return ufedmm.CollectiveVariable(name, force)


def end_to_end_distance(model):
    atoms = [(a.name, a.residue.name) for a in model.topology.atoms()]
    force = openmm.CustomBondForce("r")
    force.addBond(atoms.index(("CH3", "ACE")), atoms.index(("C", "NME")), [])
    return ufedmm.CollectiveVariable("distance", force)


def _usable(platform_name):
    try:
        platform = openmm.Platform.getPlatformByName(platform_name)
        system = openmm.System()
        system.addParticle(1.0)
        openmm.Context(system, openmm.VerletIntegrator(0.001), platform)
    except Exception:
        return False
    return True


GPU_PLATFORMS = [name for name in ("CUDA", "OpenCL") if _usable(name)]
# Usable GPU platforms as parameters marked "gpu", for pytest.mark.parametrize
GPU_PLATFORM_PARAMS = [
    pytest.param(name, marks=pytest.mark.gpu) for name in GPU_PLATFORMS
]
