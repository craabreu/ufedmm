import gc
import io

import numpy as np
import openmm
import pandas as pd
import pytest
from openmm import unit

import ufedmm


def test_state_data_reporter_with_multiple_temperatures():
    model = ufedmm.AlanineDipeptideModel()
    mass = 50 * unit.dalton * (unit.nanometer / unit.radians) ** 2
    Ks = 1000 * unit.kilojoules_per_mole / unit.radians**2
    Ts = 1500 * unit.kelvin
    s_phi = ufedmm.DynamicalVariable("s_phi", -np.pi, np.pi, mass, Ts, model.phi, Ks)
    s_psi = ufedmm.DynamicalVariable("s_psi", -np.pi, np.pi, mass, Ts, model.psi, Ks)
    ufed = ufedmm.UnifiedFreeEnergyDynamics([s_phi, s_psi], 300 * unit.kelvin)
    integrator = ufedmm.GeodesicLangevinIntegrator(
        300 * unit.kelvin, 10 / unit.picoseconds, 2 * unit.femtoseconds
    )
    platform = openmm.Platform.getPlatformByName("Reference")
    simulation = ufed.simulation(model.topology, model.system, integrator, platform)
    simulation.context.setPositions(model.positions)
    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, 1234)
    output = io.StringIO()
    reporter = ufedmm.StateDataReporter(output, 5, step=True, multipleTemperatures=True)
    simulation.reporters.append(reporter)
    simulation.step(10)

    header, *rows = output.getvalue().strip().splitlines()
    assert header == '#"Step","T[atoms] (K)","T[s_phi] (K)","T[s_psi] (K)"'
    assert len(rows) == 2
    for row in rows:
        temperatures = [float(value) for value in row.split(",")[1:]]
        assert all(np.isfinite(t) and t > 0 for t in temperatures)


def test_tee_writes_to_all_outputs_and_closes_only_its_own_files(tmp_path):
    stream = io.StringIO()
    path = tmp_path / "output.txt"
    tee = ufedmm.Tee(stream, str(path))
    print("first line", file=tee)
    tee.write("second line\n")
    tee.flush()
    del tee
    gc.collect()
    assert not stream.closed
    assert stream.getvalue() == "first line\nsecond line\n"
    assert path.read_text() == "first line\nsecond line\n"


def reporter_simulation(**kwargs):
    model = ufedmm.AlanineDipeptideModel(constraints=None)
    mass = 50 * unit.dalton * (unit.nanometer / unit.radians) ** 2
    Ks = 1000 * unit.kilojoules_per_mole / unit.radians**2
    Ts = kwargs.pop("Ts", 1500 * unit.kelvin)
    sigma = 18 * unit.degrees
    s_phi = ufedmm.DynamicalVariable(
        "s_phi", -np.pi, np.pi, mass, Ts, model.phi, Ks, sigma=sigma
    )
    s_psi = ufedmm.DynamicalVariable(
        "s_psi", -np.pi, np.pi, mass, Ts, model.psi, Ks, sigma=sigma
    )
    integrator = kwargs.pop("integrator", None) or ufedmm.MiddleMassiveNHCIntegrator(
        300 * unit.kelvin, 10 * unit.femtoseconds, 1 * unit.femtoseconds
    )
    ufed = ufedmm.UnifiedFreeEnergyDynamics([s_phi, s_psi], 300 * unit.kelvin, **kwargs)
    platform = openmm.Platform.getPlatformByName("Reference")
    simulation = ufed.simulation(model.topology, model.system, integrator, platform)
    simulation.context.setPositions(model.positions)
    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, 1234)
    return simulation


def report(simulation, steps=10, **kwargs):
    output = io.StringIO()
    simulation.reporters.append(ufedmm.StateDataReporter(output, 5, **kwargs))
    simulation.step(steps)
    header, *rows = output.getvalue().strip().splitlines()
    names = [name.strip('"') for name in header.lstrip("#").split(",")]
    values = [float(value) for value in rows[-1].split(",")]
    return names, dict(zip(names, values)), len(rows)


def driving_values(simulation):
    return [
        value
        for force in simulation.context.driving_forces
        for value in force.getCollectiveVariableValues(simulation.context)
    ]


def test_reporter_variables_come_before_trailing_columns():
    simulation = reporter_simulation()
    names, values, num_rows = report(simulation, step=True, variables=True, speed=True)
    assert num_rows == 2
    assert names == ["Step", "s_phi", "phi", "s_psi", "psi", "Speed (ns/day)"]
    expected = driving_values(simulation)
    assert [values[name] for name in names[1:5]] == pytest.approx(expected)


def test_reporter_collective_variables():
    simulation = reporter_simulation()
    names, values, _ = report(simulation, collectiveVariables=True)
    # The driving force is the only CustomCVForce in the system
    assert names == ["s_phi", "phi", "s_psi", "psi"]
    assert [values[name] for name in names] == pytest.approx(driving_values(simulation))


@pytest.mark.parametrize("bias_factor", [None, 10])
def test_reporter_hill_heights(bias_factor):
    height = 2.0 * unit.kilocalorie_per_mole
    simulation = reporter_simulation(
        height=height, frequency=5, bias_factor=bias_factor
    )
    names, values, _ = report(simulation, hillHeights=True)
    assert names == ["Height (kJ/mole)"]
    (metadynamics,) = simulation._periodic_tasks
    scaling = 1 if bias_factor is None else bias_factor / (bias_factor - 1)
    assert values["Height (kJ/mole)"] == pytest.approx(
        metadynamics.height * scaling, rel=1e-6
    )
    if bias_factor is None:
        assert metadynamics.height == pytest.approx(
            height.value_in_unit(unit.kilojoules_per_mole)
        )


def test_reporter_hill_heights_require_metadynamics():
    simulation = reporter_simulation()
    with pytest.raises(Exception, match="hillHeights"):
        report(simulation, hillHeights=True)


def test_reporter_global_parameter_states():
    simulation = reporter_simulation()
    context = simulation.context
    states = pd.DataFrame({"K_phi": [0.0, 2000.0], "K_psi": [500.0, 0.0]})
    names, values, _ = report(simulation, globalParameterStates=states)
    assert names == ["Energy[0] (kJ/mole)", "Energy[1] (kJ/mole)"]
    assert context.getParameter("K_phi") == 1000
    assert context.getParameter("K_psi") == 1000
    for index, row in states.iterrows():
        for name, value in row.items():
            context.setParameter(name, value)
        energy = context.getState(getEnergy=True).getPotentialEnergy()
        assert values[f"Energy[{index}] (kJ/mole)"] == pytest.approx(
            energy.value_in_unit(unit.kilojoules_per_mole), rel=1e-6
        )


@pytest.mark.parametrize("Ts", [300, 1500], ids=["same-temperature", "hotter"])
@pytest.mark.parametrize("regulated", [False, True], ids=["NHC", "RegulatedNHL"])
def test_reported_temperatures_exclude_extra_particles(regulated, Ts):
    # The temperature of the physical system must not account for the extra
    # particles, and each variable's temperature comes from its particle's x motion.
    temperature, tau, dt = (
        300 * unit.kelvin,
        10 * unit.femtoseconds,
        1 * unit.femtoseconds,
    )
    if regulated:
        integrator = ufedmm.RegulatedNHLIntegrator(temperature, tau, 1.0, dt, 2)
    else:
        integrator = ufedmm.MiddleMassiveNHCIntegrator(temperature, tau, dt)
    simulation = reporter_simulation(integrator=integrator, Ts=Ts * unit.kelvin)
    names, values, _ = report(simulation, temperature=True, multipleTemperatures=True)
    assert names == ["Temperature (K)", "T[atoms] (K)", "T[s_phi] (K)", "T[s_psi] (K)"]

    state = simulation.context.getState(getVelocities=True)
    v = state.getVelocities(asNumpy=True, extended=True).value_in_unit(
        unit.nanometers / unit.picosecond
    )
    system = simulation.context.getSystem()
    m = np.array([system.getParticleMass(i) / unit.dalton for i in range(len(v))])[
        :, None
    ]
    if regulated:
        c = np.sqrt(2 * np.array(integrator.getPerDofVariableByName("kTc")) / m)
        ke = 0.5 * m * v * c * np.tanh(v / c)
    else:
        ke = 0.5 * m * v**2
    natoms = len(v) - 2
    KB = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(unit.kilojoules_per_mole / unit.kelvin)
    atoms_temperature = 2 * np.sum(ke[:natoms]) / (3 * natoms * KB)
    assert values["Temperature (K)"] == pytest.approx(atoms_temperature, rel=1e-6)
    assert values["T[atoms] (K)"] == pytest.approx(atoms_temperature, rel=1e-6)
    for i, name in enumerate(["T[s_phi] (K)", "T[s_psi] (K)"]):
        assert values[name] == pytest.approx(2 * ke[natoms + i, 0] / KB, rel=1e-6)


def test_reported_density_excludes_extra_particles():
    simulation = reporter_simulation()
    names, values, _ = report(simulation, density=True)
    assert names == ["Density (g/mL)"]
    system = simulation.context.getSystem()
    natoms = system.getNumParticles() - 2
    mass = sum((system.getParticleMass(i) for i in range(natoms)), 0 * unit.dalton)
    a, b, c = simulation.context.getState().getPeriodicBoxVectors()
    density = mass / (a[0] * b[1] * c[2])
    assert values["Density (g/mL)"] == pytest.approx(
        density.value_in_unit(unit.gram / unit.item / unit.milliliter), rel=1e-6
    )


def test_multiple_temperatures_need_integrator_temperatures():
    # UnifiedFreeEnergyDynamics.simulation sets the integrator temperatures, but a
    # simulation created directly does not until its integrator first steps, so
    # reporting before that cannot split the temperatures.
    model = ufedmm.AlanineDipeptideModel(constraints=None)
    s_phi = ufedmm.DynamicalVariable("s_phi", -np.pi, np.pi, 50, 1500, model.phi, 1000)
    integrator = ufedmm.MiddleMassiveNHCIntegrator(
        300 * unit.kelvin, 10 * unit.femtoseconds, 1 * unit.femtoseconds
    )
    simulation = ufedmm.ExtendedSpaceSimulation(
        [s_phi],
        model.topology,
        model.system,
        integrator,
        openmm.Platform.getPlatformByName("Reference"),
    )
    simulation.context.setPositions(model.positions)
    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, 1234)
    reporter = ufedmm.StateDataReporter(io.StringIO(), 1, multipleTemperatures=True)
    state = simulation.context.getState(getVelocities=True, getEnergy=True)
    with pytest.raises(ValueError, match="temperatures were not set"):
        reporter.report(simulation, state)


def test_serialization_to_a_file(tmp_path):
    simulation = reporter_simulation()
    ufed = ufedmm.UnifiedFreeEnergyDynamics(
        simulation.context.variables, 300 * unit.kelvin
    )
    path = str(tmp_path / "ufed.yaml")
    ufedmm.serialize(ufed, path)
    new = ufedmm.deserialize(path)
    assert new.__getstate__().keys() == ufed.__getstate__().keys()
    assert [v.id for v in new.variables] == ["s_phi", "s_psi"]
    assert new.temperature == ufed.temperature
