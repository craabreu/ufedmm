import io

import ufedmm
from ufedmm.ufedmm import PeriodicTask

from .test_ufedmm import (
    DT,
    TAU,
    TEMP,
    reference_simulation,
    ufed_model,
)


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
