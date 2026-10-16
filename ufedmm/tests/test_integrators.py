import numpy as np
import openmm
import pytest
from openmm import unit

import ufedmm

from .common import (
    DT,
    KB,
    TAU,
    TEMP,
    nonbonded_force,
    phase_space_point,
    reference_simulation,
    ufed_model,
)

GAMMA = 10 / unit.picoseconds


def assert_same_trajectory(first, second):
    for x, y in zip(phase_space_point(first), phase_space_point(second)):
        assert x == pytest.approx(y, rel=1e-8, abs=1e-10)


@pytest.mark.parametrize(
    "options",
    [
        {"respa_loops": [3, 1]},
        {"bath_loops": 3},
        {"respa_loops": [2, 1], "bath_loops": 2, "intertwine": False},
        {"respa_loops": [2, 1], "scheme": "LF-Middle"},
    ],
    ids=["respa", "bath-loops", "not-intertwined", "LF-Middle"],
)
def test_unrolled_loops_match_while_blocks(options):
    simulations = []
    for unroll_loops in (True, False):
        model, ufed = ufed_model()
        nonbonded_force(model.system).setForceGroup(1)
        integrator = ufedmm.MiddleMassiveNHCIntegrator(
            TEMP, TAU, DT, unroll_loops=unroll_loops, **options
        )
        simulations.append(reference_simulation(model, ufed, integrator))
    for simulation in simulations:
        simulation.step(30)
    assert_same_trajectory(*simulations)


@pytest.mark.parametrize(
    "integrator_class",
    [ufedmm.MiddleMassiveNHCIntegrator, ufedmm.MiddleMassiveGGMTIntegrator],
    ids=["NHC", "GGMT"],
)
def test_embodied_force_group_is_subtracted(integrator_class):
    # Group 1 holds the inner part of the nonbonded interactions, which group 2 also
    # contains in full. With one substep per scale, kicking with f0, f1, and f2 - f1
    # must reproduce a single time-scale integration of the bonded and the full
    # nonbonded forces.
    model, ufed = ufed_model()
    for force in model.system.getForces():
        force.setForceGroup(0)
    ufedmm.add_inner_nonbonded_force(
        model.system, 0.5 * unit.nanometers, 0.6 * unit.nanometers, 1
    )
    integrator = integrator_class(
        TEMP, TAU, DT, respa_loops=[1, 1, 1], embodied_force_groups=[1]
    )
    assert integrator.getIntegrationForceGroups() == 0b101  # groups 0 and 2
    embodied = reference_simulation(model, ufed, integrator)
    model, ufed = ufed_model()
    single = reference_simulation(model, ufed, integrator_class(TEMP, TAU, DT))
    embodied.step(30)
    single.step(30)
    assert_same_trajectory(embodied, single)


def test_geodesic_langevin_with_multiple_rattles_keeps_constraints():
    model, ufed = ufed_model(constraints=openmm.app.HBonds)
    integrator = ufedmm.GeodesicLangevinIntegrator(TEMP, GAMMA, DT, num_rattles=3)
    simulation = reference_simulation(model, ufed, integrator)
    simulation.step(50)
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    x = state.getPositions(asNumpy=True).value_in_unit(unit.nanometers)
    v = state.getVelocities(asNumpy=True).value_in_unit(
        unit.nanometers / unit.picosecond
    )
    system = model.system
    assert system.getNumConstraints() > 0
    for k in range(system.getNumConstraints()):
        i, j, distance = system.getConstraintParameters(k)
        rij = x[j] - x[i]
        assert np.linalg.norm(rij) == pytest.approx(
            distance.value_in_unit(unit.nanometers), rel=1e-5
        )
        assert np.dot(rij, v[j] - v[i]) == pytest.approx(0, abs=1e-4)


@pytest.mark.parametrize(
    "make_integrator",
    [
        lambda: ufedmm.MiddleMassiveNHCIntegrator(TEMP, TAU, DT, num_rattles=1),
        lambda: ufedmm.MiddleMassiveGGMTIntegrator(TEMP, TAU, DT, num_rattles=1),
        lambda: ufedmm.RegulatedNHLIntegrator(TEMP, TAU, GAMMA, DT, 2, num_rattles=1),
        lambda: ufedmm.GeodesicLangevinIntegrator(TEMP, GAMMA, DT, scheme="Other"),
    ],
    ids=["NHC-rattle", "GGMT-rattle", "RegulatedNHL-rattle", "invalid-scheme"],
)
def test_invalid_integrator_options_are_rejected(make_integrator):
    with pytest.raises(Exception, match="constraints|scheme"):
        make_integrator()


def test_ggmt_extended_space_time_constants():
    model, ufed = ufed_model()
    taus = [0.2, 0.4]
    integrator = ufedmm.MiddleMassiveGGMTIntegrator(TEMP, TAU, DT)
    integrator.set_extended_space_time_constants(
        [tau * unit.picoseconds for tau in taus]
    )
    # The context must stay alive while the integrator's variables are read
    simulation = reference_simulation(model, ufed, integrator)  # noqa: F841
    natoms = len(model.positions)
    kT_atoms = KB * TEMP.value_in_unit(unit.kelvin)
    kT_extra = KB * ufed.variables[0].temperature
    tau = TAU.value_in_unit(unit.picoseconds)
    Q1 = np.array(integrator.getPerDofVariableByName("Q1"))
    Q2 = np.array(integrator.getPerDofVariableByName("Q2"))
    assert Q1[:natoms] == pytest.approx(np.full((natoms, 3), kT_atoms * tau**2))
    assert Q2[:natoms] == pytest.approx(
        np.full((natoms, 3), 8 / 3 * kT_atoms**3 * tau**2)
    )
    for i, tau_i in enumerate(taus):
        assert Q1[natoms + i] == pytest.approx(np.full(3, kT_extra * tau_i**2))
        assert Q2[natoms + i] == pytest.approx(
            np.full(3, 8 / 3 * kT_extra**3 * tau_i**2)
        )


@pytest.mark.parametrize("semi_regulated", [True, False], ids=["semi", "fully"])
@pytest.mark.parametrize("split", [True, False], ids=["split-OU", "unsplit-OU"])
def test_regulated_nhl_variants(semi_regulated, split):
    n = 2
    model, ufed = ufed_model()
    integrator = ufedmm.RegulatedNHLIntegrator(
        TEMP,
        TAU,
        GAMMA,
        DT,
        n,
        semi_regulated=semi_regulated,
        split_ornstein_uhlenbeck=split,
    )
    simulation = reference_simulation(model, ufed, integrator)
    simulation.step(50)
    state = simulation.context.getState(getVelocities=True, getEnergy=True)
    natoms = len(model.positions)
    masses = np.array(
        [model.system.getParticleMass(i) / unit.dalton for i in range(natoms + 2)]
    )[:, None]
    v = state.getVelocities(asNumpy=True, extended=True).value_in_unit(
        unit.nanometers / unit.picosecond
    )
    assert np.all(np.isfinite(v))
    kT = np.array(integrator.getPerDofVariableByName("kTc"))
    c = np.sqrt(n * kT / masses)
    if semi_regulated:
        expected = 0.5 * masses * v * c * np.tanh(v / c)
    else:
        expected = 0.5 * (n + 1) / n * masses * (c * np.tanh(v / c)) ** 2
    kinetic = state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole)
    assert kinetic == pytest.approx(np.sum(expected), rel=1e-6)
    # Regulated velocities never exceed c in magnitude
    assert np.all(np.abs(c * np.tanh(v / c)) < c)


def test_inner_nonbonded_force_requires_a_nonbonded_force():
    system = openmm.System()
    system.addParticle(1)
    with pytest.raises(Exception, match="requires system with NonbondedForce"):
        ufedmm.add_inner_nonbonded_force(system, 0.5, 0.6, 1)


def test_inner_nonbonded_force_rejects_parameter_offsets():
    model = ufedmm.AlanineDipeptideModel(constraints=None)
    nonbonded = nonbonded_force(model.system)
    nonbonded.addGlobalParameter("lambda", 1)
    nonbonded.addParticleParameterOffset("lambda", 0, 0.1, 0, 0)
    with pytest.raises(Exception, match="parameter offsets"):
        ufedmm.add_inner_nonbonded_force(model.system, 0.5, 0.6, 1)
