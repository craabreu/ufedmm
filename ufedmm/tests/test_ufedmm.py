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


def nested_rmsd_forces(model):
    # RMSDForce objects nested at three levels of CustomCVForce
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
    return model.system, [rmsd, cvforce]


def custom_nonbonded_force(model):
    nonbonded = nonbonded_force(model.system)
    force = openmm.CustomNonbondedForce(
        "4*epsilon*((sigma/r)^12 - (sigma/r)^6)"
        "; sigma=0.5*(sigma1+sigma2); epsilon=sqrt(epsilon1*epsilon2)"
    )
    force.addPerParticleParameter("sigma")
    force.addPerParticleParameter("epsilon")
    for index in range(nonbonded.getNumParticles()):
        _, sigma, epsilon = nonbonded.getParticleParameters(index)
        force.addParticle([sigma, epsilon])
    for index in range(nonbonded.getNumExceptions()):
        i, j, *_ = nonbonded.getExceptionParameters(index)
        force.addExclusion(i, j)
    model.system.addForce(force)
    return model.system, [force]


def custom_gb_force(model):
    forcefield = openmm.app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
    system = forcefield.createSystem(model.topology, removeCMMotion=False)
    forces = [f for f in system.getForces() if isinstance(f, openmm.CustomGBForce)]
    assert forces
    return system, forces


@pytest.mark.parametrize(
    "build",
    [nested_rmsd_forces, custom_nonbonded_force, custom_gb_force],
    ids=["nested-RMSD", "CustomNonbondedForce", "CustomGBForce"],
)
def test_extra_particles_do_not_change_physical_forces(build):
    # Forces that act on all particles must ignore the extra ones.
    model, ufed = ufed_model()
    system, tested_forces = build(model)
    for force in tested_forces:
        force.setForceGroup(1)
    plain_system = deepcopy(system)

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
    simulation = ufed.simulation(model.topology, system, integrator, platform)
    simulation.context.setPositions(positions)
    state = simulation.context.getState(getEnergy=True, getForces=True, groups={1})

    kJ_mol = unit.kilojoules_per_mole
    kJ_mol_nm = kJ_mol / unit.nanometer
    energy = state.getPotentialEnergy().value_in_unit(kJ_mol)
    assert energy != 0
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


def test_gbsaobc_force_is_rejected():
    model, ufed = ufed_model()
    model.system.addForce(openmm.GBSAOBCForce())
    integrator = ufedmm.MiddleMassiveNHCIntegrator(TEMP, TAU, DT)
    with pytest.raises(RuntimeError, match="GBSAOBCForce"):
        reference_simulation(model, ufed, integrator)


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


def test_collective_variable_id_must_be_an_identifier():
    with pytest.raises(ValueError, match="identifier"):
        ufedmm.CollectiveVariable("not valid", openmm.CustomTorsionForce("theta"))


def test_collective_variable_units():
    model = ufedmm.AlanineDipeptideModel(constraints=None)
    phi = model.phi.evaluate(model.system, model.positions)
    phi_degrees = model.phi.evaluate(model.system, model.positions, unit.degrees)
    assert phi_degrees.value_in_unit(unit.radians) == pytest.approx(phi)
    mass = model.phi.effective_mass(model.system, model.positions)
    mass_degrees = model.phi.effective_mass(model.system, model.positions, unit.degrees)
    mass_unit = unit.dalton * (unit.nanometers / unit.radians) ** 2
    assert mass_degrees.value_in_unit(mass_unit) == pytest.approx(mass)


@pytest.mark.parametrize("periodic", [True, False], ids=["periodic", "nonperiodic"])
def test_dynamical_variable_values_match_driving_force(periodic):
    # getDynamicalVariables evaluates each variable in Python, whereas the driving
    # forces evaluate the same mapping as an OpenMM expression.
    model = ufedmm.AlanineDipeptideModel(constraints=None)
    s_phi = ufedmm.DynamicalVariable(
        "s_phi", -1.0, 2.0, 50, 300, model.phi, 1000, periodic=periodic
    )
    ufed = ufedmm.UnifiedFreeEnergyDynamics([s_phi], TEMP)
    integrator = ufedmm.MiddleMassiveNHCIntegrator(TEMP, TAU, DT)
    simulation = reference_simulation(model, ufed, integrator)
    Lx = simulation.context.getParameter("Lx")
    for x in np.linspace(-1.3 * Lx, 2.1 * Lx, 17):
        simulation.context.setPositions(model.positions, [x])
        state = simulation.context.getState(getPositions=True)
        (value,) = state.getDynamicalVariables()
        (force,) = simulation.context.driving_forces
        driving_value = force.getCollectiveVariableValues(simulation.context)[0]
        assert value == pytest.approx(driving_value)
        assert -1.0 <= value <= 2.0
        fraction = x / Lx - np.floor(x / Lx)
        if periodic:
            assert value == pytest.approx(-1.0 + 3.0 * fraction)
        else:
            assert value == pytest.approx(-1.0 + 6.0 * min(fraction, 1 - fraction))


def test_many_variables_are_split_among_driving_forces():
    # A CustomCVForce takes at most 32 collective variables, and each variable here
    # needs two (itself and its colvar), so 17 variables need two driving forces.
    model = ufedmm.AlanineDipeptideModel(constraints=None)
    phi = model.phi.evaluate(model.system, model.positions)
    num_variables = 17
    Ks = np.linspace(100, 900, num_variables)
    variables = []
    for k, K in enumerate(Ks):
        colvar = ufedmm.CollectiveVariable(f"phi{k}", deepcopy(model.phi.force))
        variables.append(
            ufedmm.DynamicalVariable(f"s{k}", -np.pi, np.pi, 50, 300, colvar, K)
        )
    ufed = ufedmm.UnifiedFreeEnergyDynamics(variables, TEMP)
    integrator = ufedmm.MiddleMassiveNHCIntegrator(TEMP, TAU, DT)
    simulation = reference_simulation(model, ufed, integrator)
    context = simulation.context
    assert len(context.driving_forces) == 2
    for force in context.driving_forces:
        force.setForceGroup(30)
    context.reinitialize(preserveState=True)

    Lx = context.getParameter("Lx")
    s_values = np.linspace(-3.0, 3.0, num_variables)
    context.setPositions(model.positions, list(Lx * (s_values + np.pi) / (2 * np.pi)))
    delta = np.abs(phi - s_values)
    delta = np.minimum(delta, 2 * np.pi - delta)
    state = context.getState(getEnergy=True, groups={30})
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    assert energy == pytest.approx(np.sum(0.5 * Ks * delta**2))


def test_set_periodic_box_vectors_rescales_extra_particle_masses():
    model, ufed = ufed_model()
    integrator = ufedmm.MiddleMassiveNHCIntegrator(TEMP, TAU, DT)
    simulation = reference_simulation(model, ufed, integrator)
    context = simulation.context
    Lx = 3.0
    context.setPeriodicBoxVectors(
        openmm.Vec3(Lx, 0, 0), openmm.Vec3(0, 2.5, 0), openmm.Vec3(0, 0, 2.5)
    )
    assert context.getParameter("Lx") == pytest.approx(Lx)
    system = context.getSystem()
    natoms = model.system.getNumParticles() - 2
    for i, v in enumerate(ufed.variables):
        mass = system.getParticleMass(natoms + i).value_in_unit(unit.dalton)
        assert mass == pytest.approx(v.mass * (v._range / Lx) ** 2)
    with pytest.raises(ValueError, match="orthorhombic"):
        context.setPeriodicBoxVectors(
            openmm.Vec3(Lx, 0, 0), openmm.Vec3(0.5, 2.5, 0), openmm.Vec3(0, 0, 2.5)
        )


def test_velocities_to_temperature_with_center_of_mass_motion_remover():
    model, ufed = ufed_model()
    model.system.addForce(openmm.CMMotionRemover())
    integrator = ufedmm.MiddleMassiveNHCIntegrator(TEMP, TAU, DT)
    context = ufedmm.ExtendedSpaceContext(
        ufed.variables,
        model.system,
        integrator,
        openmm.Platform.getPlatformByName("Reference"),
    )
    context.setPositions(model.positions)
    context.setVelocitiesToTemperature(TEMP, 1234)
    natoms = len(model.positions)
    state = context.getState(getVelocities=True)
    v = state.getVelocities(asNumpy=True).value_in_unit(
        unit.nanometers / unit.picosecond
    )
    m = np.array([model.system.getParticleMass(i) / unit.dalton for i in range(natoms)])
    KB = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(unit.kilojoules_per_mole / unit.kelvin)
    temperature = np.sum(m[:, None] * v**2) / ((3 * natoms - 3) * KB)
    assert temperature == pytest.approx(TEMP.value_in_unit(unit.kelvin), rel=1e-6)


@pytest.mark.parametrize(
    "spoil, message",
    [
        (lambda m: m.system.addForce(openmm.CMMotionRemover()), "CMMotionRemover"),
        (
            lambda m: m.system.addForce(openmm.MonteCarloBarostat(1, 300)),
            "Barostat",
        ),
        (lambda m: m.topology.setPeriodicBoxVectors(None), "simulation box"),
    ],
    ids=["CMMotionRemover", "barostat", "no-box"],
)
def test_simulation_rejects_unsupported_systems(spoil, message):
    model, ufed = ufed_model()
    spoil(model)
    integrator = ufedmm.MiddleMassiveNHCIntegrator(TEMP, TAU, DT)
    with pytest.raises(Exception, match=message):
        reference_simulation(model, ufed, integrator)


def test_simulation_passes_platform_properties():
    model, ufed = ufed_model()
    integrator = ufedmm.MiddleMassiveNHCIntegrator(TEMP, TAU, DT)
    platform = openmm.Platform.getPlatformByName("CPU")
    simulation = ufed.simulation(
        model.topology, model.system, integrator, platform, {"Threads": "1"}
    )
    assert platform.getPropertyValue(simulation.context, "Threads") == "1"


def test_simulation_sets_kT_of_a_plain_custom_integrator():
    model, ufed = ufed_model()
    integrator = openmm.CustomIntegrator(DT)
    integrator.addPerDofVariable("kT", 0)
    integrator.addComputePerDof("v", "v + dt*f/m")
    integrator.addComputePerDof("x", "x + dt*v")
    simulation = reference_simulation(model, ufed, integrator)
    kT = np.array(integrator.getPerDofVariableByName("kT"))
    R = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(unit.kilojoules_per_mole / unit.kelvin)
    natoms = len(model.positions)
    assert kT[:natoms] == pytest.approx(np.full((natoms, 3), R * 300))
    assert kT[natoms:] == pytest.approx(np.full((2, 3), R * 1500))
    simulation.step(5)
