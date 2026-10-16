import io

import pytest
from openmm import unit

import ufedmm
from ufedmm.ufedmm import PeriodicTask

from .common import (
    DT,
    METAD_GROUP,
    TAU,
    TEMP,
    phase_space_point,
    reference_simulation,
    ufed_model,
)


def metadynamics_simulation(enforce_gridless, bias_psi=True):
    model, ufed = ufed_model(
        height=2.0 * unit.kilocalorie_per_mole,
        enforce_gridless=enforce_gridless,
        bias_psi=bias_psi,
    )
    integrator = ufedmm.MiddleMassiveNHCIntegrator(TEMP, TAU, DT)
    simulation = reference_simulation(model, ufed, integrator)
    simulation._periodic_tasks[0].force.setForceGroup(METAD_GROUP)
    simulation.context.reinitialize(preserveState=True)
    return simulation


def bias_energy(simulation):
    state = simulation.context.getState(getEnergy=True, groups={METAD_GROUP})
    return state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)


@pytest.mark.parametrize(
    "enforce_gridless, bias_psi",
    [(False, False), (False, True), (True, True)],
    ids=["gridded-1D", "gridded-2D", "gridless-2D"],
)
@pytest.mark.parametrize("target", ["path", "stream"])
@pytest.mark.parametrize("prior_steps", [0, 80], ids=["fresh", "ahead"])
def test_checkpoint_restores_bias_and_continuation(
    enforce_gridless, bias_psi, target, prior_steps, tmp_path
):
    original = metadynamics_simulation(enforce_gridless, bias_psi)
    original.step(50)
    expected_bias = bias_energy(original)
    expected_point = phase_space_point(original)
    checkpoint = tmp_path / "checkpoint.bin" if target == "path" else io.BytesIO()
    original.saveCheckpoint(str(checkpoint) if target == "path" else checkpoint)
    original.step(30)

    # A fresh simulation must make room for the restored hills, and one that has
    # deposited more hills than the checkpoint holds must discard the extra ones.
    restored = metadynamics_simulation(enforce_gridless, bias_psi)
    restored.step(prior_steps)
    if target == "stream":
        checkpoint.seek(0)
    restored.loadCheckpoint(str(checkpoint) if target == "path" else checkpoint)
    assert restored.currentStep == 50
    assert bias_energy(restored) == pytest.approx(expected_bias, rel=1e-12)
    for x, y in zip(phase_space_point(restored), expected_point):
        assert x == pytest.approx(y, rel=1e-12, abs=1e-14)

    restored.step(30)
    assert bias_energy(restored) == pytest.approx(bias_energy(original), rel=1e-12)
    for x, y in zip(phase_space_point(restored), phase_space_point(original)):
        assert x == pytest.approx(y, rel=1e-12, abs=1e-14)


@pytest.mark.parametrize("enforce_gridless", [False, True], ids=["gridded", "gridless"])
def test_truncated_checkpoint_is_rejected(enforce_gridless):
    simulation = metadynamics_simulation(enforce_gridless)
    simulation.step(20)
    checkpoint = io.BytesIO()
    simulation.saveCheckpoint(checkpoint)
    truncated = io.BytesIO(checkpoint.getvalue()[:12])
    with pytest.raises(ValueError, match="end of checkpoint"):
        metadynamics_simulation(enforce_gridless).loadCheckpoint(truncated)


class StepRecorder(PeriodicTask):
    def __init__(self, frequency):
        super().__init__(frequency)
        self.steps = []

    def report(self, simulation, state):
        self.steps.append(simulation.currentStep)


def test_user_defined_periodic_task():
    model, ufed = ufed_model(height=None)
    integrator = ufedmm.MiddleMassiveNHCIntegrator(TEMP, TAU, DT)
    simulation = reference_simulation(model, ufed, integrator)
    task = StepRecorder(5)
    simulation.add_periodic_task(task)
    simulation.step(20)
    assert task.steps == [5, 10, 15, 20]
    checkpoint = io.BytesIO()
    simulation.saveCheckpoint(checkpoint)
    checkpoint.seek(0)
    simulation.loadCheckpoint(checkpoint)
    assert simulation.currentStep == 20
