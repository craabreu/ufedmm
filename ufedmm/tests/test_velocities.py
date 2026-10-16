import numpy as np
import openmm
import pytest
from openmm import unit

import ufedmm

from .common import KB


def ufed_simulation(integrator, constraints=openmm.app.HBonds, massless_atom=None):
    model = ufedmm.AlanineDipeptideModel(constraints=constraints)
    if massless_atom is not None:
        model.system.setParticleMass(massless_atom, 0)
    mass = 50 * unit.dalton * (unit.nanometer / unit.radians) ** 2
    Ks = 1000 * unit.kilojoules_per_mole / unit.radians**2
    Ts = 1500 * unit.kelvin
    s_phi = ufedmm.DynamicalVariable("s_phi", -np.pi, np.pi, mass, Ts, model.phi, Ks)
    s_psi = ufedmm.DynamicalVariable("s_psi", -np.pi, np.pi, mass, Ts, model.psi, Ks)
    ufed = ufedmm.UnifiedFreeEnergyDynamics([s_phi, s_psi], 300 * unit.kelvin)
    platform = openmm.Platform.getPlatformByName("Reference")
    simulation = ufed.simulation(model.topology, model.system, integrator, platform)
    simulation.context.setPositions(model.positions)
    return model, simulation


def langevin():
    return ufedmm.GeodesicLangevinIntegrator(
        300 * unit.kelvin, 10 / unit.picoseconds, 1 * unit.femtoseconds
    )


def test_thermostat_targets_only_x_for_extra_particles():
    model, simulation = ufed_simulation(langevin())
    natoms = model.topology.getNumAtoms()
    kT = np.array(simulation.integrator.getPerDofVariableByName("kT"))
    assert kT[:natoms] == pytest.approx(np.full((natoms, 3), KB * 300))
    assert kT[natoms:] == pytest.approx(np.array([[KB * 1500, 0, 0]] * 2))


@pytest.mark.parametrize(
    "integrator_factory",
    [
        lambda: ufedmm.MiddleMassiveNHCIntegrator(
            300 * unit.kelvin, 100 * unit.femtoseconds, 1 * unit.femtoseconds
        ),
        lambda: ufedmm.MiddleMassiveGGMTIntegrator(
            300 * unit.kelvin, 100 * unit.femtoseconds, 1 * unit.femtoseconds
        ),
        lambda: ufedmm.RegulatedNHLIntegrator(
            300 * unit.kelvin,
            100 * unit.femtoseconds,
            1 / unit.picoseconds,
            1 * unit.femtoseconds,
            2,
        ),
    ],
    ids=["NHC", "GGMT", "RegulatedNHL"],
)
def test_massive_thermostats_stay_finite(integrator_factory):
    integrator = integrator_factory()
    _, simulation = ufed_simulation(integrator, constraints=None)
    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, 1234)
    simulation.step(50)
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    assert np.all(np.isfinite(state.getPositions(asNumpy=True)._value))
    assert np.all(np.isfinite(state.getVelocities(asNumpy=True)._value))


def atom_velocities(model, simulation):
    state = simulation.context.getState(getVelocities=True)
    natoms = model.topology.getNumAtoms()
    all_velocities = state.getVelocities(asNumpy=True, extended=True)
    v = all_velocities.value_in_unit(unit.nanometers / unit.picosecond)
    masses = np.array(
        [model.system.getParticleMass(i) / unit.dalton for i in range(natoms)]
    )
    return masses, v[:natoms], v[natoms:]


def test_set_velocities_to_temperature():
    model, simulation = ufed_simulation(langevin())
    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, 1234)
    masses, v, vxs = atom_velocities(model, simulation)

    # No center-of-mass motion of the physical system
    assert np.sum(masses[:, None] * v, axis=0) == pytest.approx(np.zeros(3), abs=1e-8)

    # Instantaneous temperature exactly at the target
    dof = 3 * len(masses) - model.system.getNumConstraints()
    temperature = np.sum(masses[:, None] * v**2) / (dof * KB)
    assert temperature == pytest.approx(300, rel=1e-6)

    # Velocities satisfy the bond constraints
    positions = model.positions.value_in_unit(unit.nanometers)
    for k in range(model.system.getNumConstraints()):
        i, j, _ = model.system.getConstraintParameters(k)
        rij = np.array(positions[j]) - np.array(positions[i])
        assert np.dot(rij, v[j] - v[i]) == pytest.approx(0, abs=1e-6)

    # Extra particles move along x only
    assert np.all(vxs[:, 0] != 0)
    assert vxs[:, 1:] == pytest.approx(np.zeros((2, 2)))


def test_set_velocities_to_temperature_with_massless_particle():
    model, simulation = ufed_simulation(langevin(), constraints=None, massless_atom=0)
    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, 1234)
    masses, v, _ = atom_velocities(model, simulation)
    assert np.all(np.isfinite(v))
    assert v[0] == pytest.approx(np.zeros(3))
    dof = 3 * np.count_nonzero(masses)
    temperature = np.sum(masses[:, None] * v**2) / (dof * KB)
    assert temperature == pytest.approx(300, rel=1e-6)


def test_get_velocities_split():
    model, simulation = ufed_simulation(langevin())
    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, 1234)
    state = simulation.context.getState(getVelocities=True)
    natoms = model.topology.getNumAtoms()
    assert len(state.getVelocities()) == natoms
    # Backward compatible: a single list of all velocities by default
    assert len(state.getVelocities(extended=True)) == natoms + 2
    atoms, extra = state.getVelocities(extended=True, split=True)
    assert (len(atoms), len(extra)) == (natoms, 2)


def test_regulated_kinetic_energy_before_first_step():
    model = ufedmm.AlanineDipeptideModel(constraints=None)
    integrator = ufedmm.RegulatedNHLIntegrator(
        300 * unit.kelvin,
        100 * unit.femtoseconds,
        1 / unit.picoseconds,
        1 * unit.femtoseconds,
        2,
    )
    context = openmm.Context(
        model.system, integrator, openmm.Platform.getPlatformByName("Reference")
    )
    context.setPositions(model.positions)
    context.setVelocitiesToTemperature(300 * unit.kelvin, 1234)
    state = context.getState(getEnergy=True, getVelocities=True)
    m = np.array(
        [
            model.system.getParticleMass(i) / unit.dalton
            for i in range(len(model.positions))
        ]
    )[:, None]
    v = state.getVelocities(asNumpy=True).value_in_unit(
        unit.nanometers / unit.picosecond
    )
    c = np.sqrt(2 * KB * 300 / m)
    expected = np.sum(0.5 * m * v * c * np.tanh(v / c))
    kinetic = state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole)
    assert kinetic == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("Ts", [300, 1500], ids=["same-temperature", "hotter"])
@pytest.mark.parametrize(
    "make_integrator",
    [
        lambda: ufedmm.MiddleMassiveNHCIntegrator(
            300 * unit.kelvin, 100 * unit.femtoseconds, 1 * unit.femtoseconds
        ),
        langevin,
    ],
    ids=["NHC", "GeodesicLangevin"],
)
def test_extra_particles_move_along_x_only(make_integrator, Ts):
    model = ufedmm.AlanineDipeptideModel(constraints=None)
    mass = 50 * unit.dalton * (unit.nanometer / unit.radians) ** 2
    Ks = 1000 * unit.kilojoules_per_mole / unit.radians**2
    s_phi = ufedmm.DynamicalVariable("s_phi", -np.pi, np.pi, mass, Ts, model.phi, Ks)
    s_psi = ufedmm.DynamicalVariable("s_psi", -np.pi, np.pi, mass, Ts, model.psi, Ks)
    ufed = ufedmm.UnifiedFreeEnergyDynamics([s_phi, s_psi], 300 * unit.kelvin)
    integrator = make_integrator()
    platform = openmm.Platform.getPlatformByName("Reference")
    simulation = ufed.simulation(model.topology, model.system, integrator, platform)
    natoms = model.topology.getNumAtoms()
    kT = np.array(integrator.getPerDofVariableByName("kT"))
    assert kT[natoms:] == pytest.approx(np.array([[KB * Ts, 0, 0]] * 2))
    simulation.context.setPositions(model.positions)
    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, 1234)
    simulation.step(100)
    state = simulation.context.getState(getVelocities=True)
    v = state.getVelocities(asNumpy=True, extended=True).value_in_unit(
        unit.nanometers / unit.picosecond
    )
    assert np.all(v[natoms:, 0] != 0)
    assert np.all(v[natoms:, 1:] == 0)
