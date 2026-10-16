"""Physical checks: canonical sampling, energy conservation, and agreement between
OpenMM platforms."""

import numpy as np
import openmm
import pytest
from openmm import unit

import ufedmm

from .common import GPU_PLATFORM_PARAMS, GPU_PLATFORMS, KB, end_to_end_distance

# A bath of independent harmonic oscillators. Like the atoms of a UFED system, the
# first NUM_ATOMS particles are thermostatted at T in all directions. Like its extra
# particles, the last NUM_EXTRA are thermostatted at TS along x only.
NUM_ATOMS, NUM_EXTRA = 200, 100
MASS, STIFFNESS = 10.0, 4000.0
T, TS = 300.0, 1500.0
DT = 2 * unit.femtoseconds
TAU = 0.1 * unit.picoseconds
GAMMA = 10 / unit.picoseconds
REGULATION = 2


def oscillator_bath(integrator, seed=1):
    integrator.setRandomNumberSeed(seed)
    system = openmm.System()
    force = openmm.CustomExternalForce(f"{0.5 * STIFFNESS}*(x^2 + y^2 + z^2)")
    for i in range(NUM_ATOMS + NUM_EXTRA):
        system.addParticle(MASS)
        force.addParticle(i, [])
    system.addForce(force)
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(system, integrator, platform)
    integrator.update_temperatures(T * unit.kelvin, [TS * unit.kelvin] * NUM_EXTRA)
    # Start from canonical positions and velocities. Starting at rest would leave a
    # slowly relaxing transient common to all oscillators.
    rng = np.random.default_rng(seed)
    positions = np.zeros((NUM_ATOMS + NUM_EXTRA, 3))
    velocities = np.zeros((NUM_ATOMS + NUM_EXTRA, 3))
    for dofs, temperature in ((np.s_[:NUM_ATOMS, :], T), (np.s_[NUM_ATOMS:, 0], TS)):
        shape = positions[dofs].shape
        positions[dofs] = rng.normal(0, np.sqrt(KB * temperature / STIFFNESS), shape)
        velocities[dofs] = rng.normal(0, np.sqrt(KB * temperature / MASS), shape)
    context.setPositions(positions)
    context.setVelocities(velocities)
    return context


def kinetic_energies(integrator, v, regulated):
    if regulated:
        kT = np.array(integrator.getPerDofVariableByName("kTc"))
        c = np.sqrt(REGULATION * kT / MASS)
        return 0.5 * MASS * v * c * np.tanh(v / c)
    return 0.5 * MASS * v**2


@pytest.mark.parametrize(
    "make_integrator, regulated",
    [
        (lambda: ufedmm.MiddleMassiveNHCIntegrator(T, TAU, DT), False),
        (lambda: ufedmm.MiddleMassiveGGMTIntegrator(T, TAU, DT), False),
        (lambda: ufedmm.GeodesicLangevinIntegrator(T, GAMMA, DT), False),
        (lambda: ufedmm.RegulatedNHLIntegrator(T, TAU, GAMMA, DT, REGULATION), True),
    ],
    ids=["NHC", "GGMT", "GeodesicLangevin", "RegulatedNHL"],
)
@pytest.mark.slow
def test_thermostats_sample_each_temperature(make_integrator, regulated):
    # Every thermostatted degree of freedom must sample the canonical distribution at
    # its own temperature. Mean kinetic and potential energies of kT/2 are enforced by
    # any feedback thermostat, even a wrong one, so the fourth moments of positions
    # (and of velocities, unless they are regulated) are checked as well. The error
    # of each average is estimated from the spread of the time averages of
    # independent oscillators.
    integrator = make_integrator()
    context = oscillator_bath(integrator)
    integrator.step(2000)
    shape = (NUM_ATOMS + NUM_EXTRA, 3)
    sums = {name: np.zeros(shape) for name in ("kinetic", "potential", "v4", "x4")}
    num_samples = 1000
    for _ in range(num_samples):
        integrator.step(10)
        state = context.getState(getPositions=True, getVelocities=True)
        x = state.getPositions(asNumpy=True).value_in_unit(unit.nanometers)
        v = state.getVelocities(asNumpy=True).value_in_unit(
            unit.nanometers / unit.picosecond
        )
        sums["kinetic"] += kinetic_energies(integrator, v, regulated)
        sums["potential"] += 0.5 * STIFFNESS * x**2
        sums["v4"] += v**4
        sums["x4"] += x**4
    averages = {name: total / num_samples for name, total in sums.items()}

    groups = {
        "atoms": (np.s_[:NUM_ATOMS, :], T),
        "extra along x": (np.s_[NUM_ATOMS:, 0], TS),
    }
    for group, (dofs, temperature) in groups.items():
        kT = KB * temperature
        expected = {
            "kinetic": 0.5 * kT,
            "potential": 0.5 * kT,
            "x4": 3 * (kT / STIFFNESS) ** 2,
        }
        if not regulated:
            expected["v4"] = 3 * (kT / MASS) ** 2
        for name, value in expected.items():
            ratio = averages[name][dofs].ravel() / value
            error = ratio.std(ddof=1) / np.sqrt(ratio.size)
            assert abs(ratio.mean() - 1) < 4 * error, f"{name} of {group}"
            assert error < 0.03
    # The extra particles start at rest along y and z and must stay that way
    assert np.max(averages["kinetic"][NUM_ATOMS:, 1:]) < 1e-12


def ufed_variables(model):
    """A periodic torsion and a non-periodic distance, both biased."""
    mass = 50 * unit.dalton * (unit.nanometer / unit.radians) ** 2
    Ks = 1000 * unit.kilojoules_per_mole / unit.radians**2
    distance = end_to_end_distance(model)
    d0 = distance.evaluate(model.system, model.positions)
    return [
        ufedmm.DynamicalVariable(
            "s_phi", -np.pi, np.pi, mass, 1500, model.phi, Ks, sigma=0.3
        ),
        ufedmm.DynamicalVariable(
            "s_d",
            d0 - 0.25,
            d0 + 0.25,
            50,
            1500,
            distance,
            1000,
            periodic=False,
            sigma=0.025,
        ),
    ]


def energy_fluctuation(step_size, respa_loops):
    model = ufedmm.AlanineDipeptideModel(constraints=None)
    if respa_loops:
        nonbonded = next(
            f for f in model.system.getForces() if isinstance(f, openmm.NonbondedForce)
        )
        nonbonded.setForceGroup(1)
    # Without a bath, the middle scheme is plain velocity Verlet (NVE)
    integrator = ufedmm.integrators.AbstractMiddleRespaIntegrator(
        T, step_size, respa_loops=respa_loops
    )
    context = ufedmm.ExtendedSpaceContext(
        ufed_variables(model),
        model.system,
        integrator,
        openmm.Platform.getPlatformByName("Reference"),
    )
    context.setPositions(model.positions)
    context.setVelocitiesToTemperature(T * unit.kelvin, 1)
    num_steps = round(2 * unit.picoseconds / step_size)
    energies = []
    for _ in range(400):
        integrator.step(num_steps // 400)
        state = context.getState(getEnergy=True)
        energy = state.getKineticEnergy() + state.getPotentialEnergy()
        energies.append(energy.value_in_unit(unit.kilojoules_per_mole))
    return np.std(energies)


@pytest.mark.parametrize(
    "step_size, respa_loops",
    [(0.5 * unit.femtoseconds, None), (2 * unit.femtoseconds, [4, 1])],
    ids=["single-time-scale", "RESPA"],
)
def test_extended_system_energy_error_is_second_order(step_size, respa_loops):
    # Without a thermostat, the extended system (atoms, extra particles, and the
    # driving potentials) must conserve energy with fluctuations of order dt^2, so
    # halving the step size divides them by about four. A driving force inconsistent
    # with its energy would break this.
    coarse = energy_fluctuation(step_size, respa_loops)
    fine = energy_fluctuation(step_size / 2, respa_loops)
    assert coarse < 0.1
    assert 3.2 < coarse / fine < 5


def platform_trajectory(platform_name):
    model = ufedmm.AlanineDipeptideModel(constraints=None)
    ufed = ufedmm.UnifiedFreeEnergyDynamics(
        ufed_variables(model), T, height=2.0, frequency=10
    )
    integrator = ufedmm.MiddleMassiveNHCIntegrator(T, TAU, DT / 2)
    platform = openmm.Platform.getPlatformByName(platform_name)
    properties = {"Precision": "mixed"} if platform_name in GPU_PLATFORMS else None
    simulation = ufed.simulation(
        model.topology, model.system, integrator, platform, properties
    )
    simulation.context.setPositions(model.positions)
    simulation.context.setVelocitiesToTemperature(T * unit.kelvin, 1234)
    simulation.step(30)
    state = simulation.context.getState(
        getEnergy=True, getForces=True, getPositions=True
    )
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    forces = state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoules_per_mole / unit.nanometer
    )
    positions = state.getPositions(asNumpy=True, extended=True, split=False)
    return energy, forces, positions.value_in_unit(unit.nanometers)


@pytest.mark.parametrize("platform_name", ["CPU", *GPU_PLATFORM_PARAMS])
def test_platforms_agree_with_reference(platform_name):
    # After 30 steps with three hills deposited, every platform must follow the
    # Reference trajectory.
    energy, forces, positions = platform_trajectory(platform_name)
    ref_energy, ref_forces, ref_positions = platform_trajectory("Reference")
    assert energy == pytest.approx(ref_energy, rel=1e-4)
    assert np.linalg.norm(forces - ref_forces) < 1e-4 * np.linalg.norm(ref_forces)
    assert positions == pytest.approx(ref_positions, abs=1e-6)
