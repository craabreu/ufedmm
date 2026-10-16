import gc
import io

import numpy as np
import openmm
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
