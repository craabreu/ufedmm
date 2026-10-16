"""
Unit and regression test for the ufedmm package.
"""

import io
from copy import deepcopy

import numpy as np
import openmm
import pytest
from openmm import unit

import ufedmm

TEMP = 300 * unit.kelvin
TAU = 10 * unit.femtoseconds
DT = 1 * unit.femtoseconds
METAD_GROUP = 31
KB = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(unit.kilojoules_per_mole / unit.kelvin)


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


def full_state(obj):
    """Everything that defines a serializable ufedmm object, with forces as XML."""
    if isinstance(obj, openmm.Force):
        return openmm.XmlSerializer.serialize(obj)
    if isinstance(obj, ufedmm.CollectiveVariable):
        return {"id": obj.id, "force": full_state(obj.force)}
    if isinstance(obj, (ufedmm.UnifiedFreeEnergyDynamics, ufedmm.DynamicalVariable)):
        return full_state(obj.__getstate__())
    if isinstance(obj, (list, tuple)):
        return [full_state(item) for item in obj]
    if isinstance(obj, dict):
        return {key: full_state(value) for key, value in obj.items()}
    return obj


def test_serialization():
    _, old = ufed_model(height=2.0 * unit.kilocalorie_per_mole, bias_factor=10)
    pipe = io.StringIO()
    ufedmm.serialize(old, pipe)
    pipe.seek(0)
    new = ufedmm.deserialize(pipe)
    assert isinstance(new, ufedmm.UnifiedFreeEnergyDynamics)
    assert full_state(new) == full_state(old)


def test_variables():
    model, ufed = ufed_model()
    integrator = ufedmm.MiddleMassiveNHCIntegrator(TEMP, TAU, DT)
    simulation = reference_simulation(model, ufed, integrator)
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
    simulation.context.setPositions(*state.getPositions(extended=True))
    xvars = simulation.context.getState(getPositions=True).getDynamicalVariables()
    assert cvs[0] == pytest.approx(xvars[0])
    assert cvs[2] == pytest.approx(xvars[1])


THERMOSTATS = pytest.mark.parametrize(
    "integrator_class",
    [ufedmm.MiddleMassiveNHCIntegrator, ufedmm.MiddleMassiveGGMTIntegrator],
    ids=["NHC", "GGMT"],
)


@THERMOSTATS
def test_respa_with_single_loops_matches_single_time_scale(integrator_class):
    # With respa_loops=[1, 1], splitting the forces into two groups only reorders
    # the half kicks, so the trajectory must be that of a single time scale.
    model, ufed = ufed_model()
    nonbonded_force(model.system).setForceGroup(1)
    respa = reference_simulation(
        model, ufed, integrator_class(TEMP, TAU, DT, respa_loops=[1, 1])
    )
    model, ufed = ufed_model()
    single = reference_simulation(model, ufed, integrator_class(TEMP, TAU, DT))
    respa.step(50)
    single.step(50)
    for x, y in zip(phase_space_point(respa), phase_space_point(single)):
        assert x == pytest.approx(y, rel=1e-8, abs=1e-10)


@THERMOSTATS
def test_respa_with_empty_outer_group_matches_inner_step_size(integrator_class):
    # When all forces are in the inner group, one outer step of size dt is the same
    # as n steps of size dt/n, because consecutive inner half kicks merge into one.
    n = 5
    model, ufed = ufed_model()
    respa = reference_simulation(
        model, ufed, integrator_class(TEMP, TAU, n * DT, respa_loops=[n, 1])
    )
    model, ufed = ufed_model()
    single = reference_simulation(model, ufed, integrator_class(TEMP, TAU, DT))
    respa.step(10)
    single.step(10 * n)
    for x, y in zip(phase_space_point(respa), phase_space_point(single)):
        assert x == pytest.approx(y, rel=1e-8, abs=1e-10)


class HillRecorder:
    """Records the center and height of each hill as it is deposited.

    Periodic tasks report before regular reporters and receive the same state, so
    the dynamical variables seen here are the centers of the hills just deposited.
    """

    def __init__(self, metadynamics):
        self._metadynamics = metadynamics
        self.centers = []
        self.heights = []

    def describeNextReport(self, simulation):
        return self._metadynamics.describeNextReport(simulation)

    def report(self, simulation, state):
        self.centers.append(state.getDynamicalVariables())
        self.heights.append(self._metadynamics.height)


def von_mises_bias(variables, centers, heights, point):
    """Sum of von Mises hills, which is what ufedmm deposits for periodic variables."""
    exponents = 0.0
    for i, v in enumerate(variables):
        if v.sigma is None:
            continue
        factor = 2 * np.pi / v._range
        delta = point[i] - np.asarray(centers)[:, i]
        exponents += (np.cos(factor * delta) - 1) / (factor * v.sigma) ** 2
    return np.sum(np.asarray(heights) * np.exp(exponents))


def run_metadynamics(steps, **kwargs):
    height = 2.0 * unit.kilocalorie_per_mole
    model, ufed = ufed_model(height=height, **kwargs)
    integrator = ufedmm.MiddleMassiveNHCIntegrator(TEMP, TAU, DT)
    simulation = reference_simulation(model, ufed, integrator)
    (metadynamics,) = simulation._periodic_tasks
    metadynamics.force.setForceGroup(METAD_GROUP)
    simulation.context.reinitialize(preserveState=True)
    recorder = HillRecorder(metadynamics)
    simulation.reporters.append(recorder)
    simulation.step(steps)
    state = simulation.context.getState(
        getPositions=True, getEnergy=True, groups={METAD_GROUP}
    )
    bias = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    return ufed, recorder, state.getDynamicalVariables(), bias


@pytest.mark.parametrize(
    "enforce_gridless, bias_psi, tolerance",
    [
        pytest.param(False, False, 1e-4, id="gridded-1D"),
        pytest.param(False, True, 1e-4, id="gridded-2D"),
        pytest.param(True, False, 1e-10, id="gridless-1D"),
        pytest.param(True, True, 1e-10, id="gridless-2D"),
    ],
)
def test_metadynamics_bias_is_a_sum_of_hills(enforce_gridless, bias_psi, tolerance):
    ufed, recorder, point, bias = run_metadynamics(
        100, enforce_gridless=enforce_gridless, bias_psi=bias_psi
    )
    height = ufed.height
    assert len(recorder.centers) == 10
    assert recorder.heights == pytest.approx([height] * 10)
    expected = von_mises_bias(ufed.variables, recorder.centers, recorder.heights, point)
    assert bias == pytest.approx(expected, rel=tolerance)


@pytest.mark.parametrize(
    "enforce_gridless, bias_psi, tolerance",
    [
        pytest.param(False, False, 1e-4, id="gridded-1D"),
        pytest.param(False, True, 1e-4, id="gridded-2D"),
        pytest.param(True, False, 1e-10, id="gridless-1D"),
        pytest.param(True, True, 1e-10, id="gridless-2D"),
    ],
)
def test_well_tempered_metadynamics_scales_hill_heights(
    enforce_gridless, bias_psi, tolerance
):
    bias_factor = 10
    ufed, recorder, point, bias = run_metadynamics(
        100,
        bias_factor=bias_factor,
        enforce_gridless=enforce_gridless,
        bias_psi=bias_psi,
    )
    variables = ufed.variables
    delta_kT = (bias_factor - 1) * KB * variables[0].temperature
    expected_heights = [ufed.height]
    for k in range(1, len(recorder.centers)):
        previous = von_mises_bias(
            variables,
            recorder.centers[:k],
            recorder.heights[:k],
            recorder.centers[k],
        )
        expected_heights.append(ufed.height * np.exp(-previous / delta_kT))
    assert recorder.heights == pytest.approx(expected_heights, rel=tolerance)
    assert recorder.heights[-1] < 0.9 * ufed.height
    expected = von_mises_bias(variables, recorder.centers, recorder.heights, point)
    assert bias == pytest.approx(expected, rel=tolerance)


def test_rmsd_forces():
    # RMSDForce objects nested at three levels of CustomCVForce must still act only
    # on the physical particles once the extra particles are added.
    model, ufed = ufed_model()
    rmsd = openmm.RMSDForce(  # Level 1
        model.positions, np.arange(model.system.getNumParticles())
    )
    cvforce = openmm.CustomCVForce("rmsd + cvforce")
    cvforce.addCollectiveVariable("rmsd", deepcopy(rmsd))  # Level 2
    inner_cvforce = openmm.CustomCVForce("rmsd")
    inner_cvforce.addCollectiveVariable("rmsd", deepcopy(rmsd))  # Level 3
    cvforce.addCollectiveVariable("cvforce", inner_cvforce)
    for force in (rmsd, cvforce):
        force.setForceGroup(1)
        model.system.addForce(force)
    plain_system = deepcopy(model.system)

    rng = np.random.default_rng(1234)
    displacements = rng.normal(scale=0.01, size=(len(model.positions), 3))
    positions = [
        position + openmm.Vec3(*delta) * unit.nanometers
        for position, delta in zip(model.positions, displacements)
    ]
    platform = openmm.Platform.getPlatformByName("Reference")
    plain = openmm.Context(plain_system, openmm.VerletIntegrator(DT), platform)
    plain.setPositions(positions)
    expected = plain.getState(getEnergy=True, getForces=True, groups={1})

    integrator = ufedmm.MiddleMassiveNHCIntegrator(TEMP, TAU, DT)
    simulation = ufed.simulation(model.topology, model.system, integrator, platform)
    simulation.context.setPositions(positions)
    state = simulation.context.getState(getEnergy=True, getForces=True, groups={1})

    kJ_mol = unit.kilojoules_per_mole
    kJ_mol_nm = kJ_mol / unit.nanometer
    energy = state.getPotentialEnergy().value_in_unit(kJ_mol)
    assert energy > 0
    assert energy == pytest.approx(expected.getPotentialEnergy().value_in_unit(kJ_mol))
    forces = state.getForces(asNumpy=True).value_in_unit(kJ_mol_nm)
    natoms = len(model.positions)
    assert forces[:natoms] == pytest.approx(
        expected.getForces(asNumpy=True).value_in_unit(kJ_mol_nm)
    )
    assert forces[natoms:] == pytest.approx(np.zeros((2, 3)))

    simulation.context.setVelocitiesToTemperature(TEMP, 1234)
    simulation.step(10)
    state = simulation.context.getState(getPositions=True)
    assert np.all(np.isfinite(state.getPositions(asNumpy=True)._value))


@pytest.mark.parametrize("form", ["vec3-list", "numpy", "numpy-quantity"])
def test_set_positions_accepts_numpy_arrays(form):
    model, ufed = ufed_model()
    integrator = ufedmm.MiddleMassiveNHCIntegrator(TEMP, TAU, DT)
    simulation = reference_simulation(model, ufed, integrator)
    expected = simulation.context.getState(getPositions=True)
    positions = {
        "vec3-list": model.positions,
        "numpy": np.array(model.positions.value_in_unit(unit.nanometers)),
        "numpy-quantity": expected.getPositions(asNumpy=True),
    }[form]
    simulation.context.setPositions(positions)
    state = simulation.context.getState(getPositions=True)
    assert state.getDynamicalVariables() == pytest.approx(
        expected.getDynamicalVariables()
    )
    assert state.getPositions(asNumpy=True)._value == pytest.approx(
        expected.getPositions(asNumpy=True)._value
    )
