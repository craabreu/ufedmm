import numpy as np
import openmm
import pytest
from openmm import unit

import ufedmm

from .test_ufedmm import DT, TAU, TEMP, reference_simulation

KB = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(unit.kilojoules_per_mole / unit.kelvin)
METAD_GROUP = 31
HEIGHT = 2.0 * unit.kilocalorie_per_mole
MASS = 50 * unit.dalton * (unit.nanometer / unit.radians) ** 2
KS = 1000 * unit.kilojoules_per_mole / unit.radians**2
TS = 1500 * unit.kelvin


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


def metadynamics_model(specs, temperatures=None, **kwargs):
    """A UFED setup with one dynamical variable per spec.

    Each spec is a tuple ``(colvar, sigma)``, where ``colvar`` is "phi", "psi",
    "omega" (periodic torsions spanning 360 degrees), or "distance" (a non-periodic
    end-to-end distance spanning 0.5 nm around its initial value), and ``sigma`` is a
    fraction of the variable's range (``None`` means no bias).
    """
    model = ufedmm.AlanineDipeptideModel(constraints=None)
    colvars = {
        "phi": model.phi,
        "psi": model.psi,
        "omega": torsion(
            model,
            "omega",
            [("CA", "ALA"), ("C", "ALA"), ("N", "NME"), ("C", "NME")],
        ),
        "distance": end_to_end_distance(model),
    }
    temperatures = temperatures or [TS] * len(specs)
    variables = []
    for (name, sigma), temperature in zip(specs, temperatures):
        cv = colvars[name]
        if name == "distance":
            center = cv.evaluate(model.system, model.positions)
            limits = (center - 0.25, center + 0.25)
            mass, force_constant = 50, 1000
        else:
            limits = (-np.pi, np.pi)
            mass, force_constant = MASS, KS
        variables.append(
            ufedmm.DynamicalVariable(
                f"s_{name}",
                *limits,
                mass,
                temperature,
                cv,
                force_constant,
                periodic=name != "distance",
                sigma=None if sigma is None else sigma * (limits[1] - limits[0]),
            )
        )
    ufed = ufedmm.UnifiedFreeEnergyDynamics(variables, TEMP, HEIGHT, 10, **kwargs)
    return model, ufed


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


def hills_bias(variables, centers, heights, point):
    """Sum of the hills ufedmm deposits: von Mises for periodic variables and
    Gaussian for non-periodic ones."""
    exponents = 0.0
    for i, v in enumerate(variables):
        if v.sigma is None:
            continue
        delta = point[i] - np.asarray(centers)[:, i]
        if v.periodic:
            factor = 2 * np.pi / v._range
            exponents += (np.cos(factor * delta) - 1) / (factor * v.sigma) ** 2
        else:
            exponents += -0.5 * (delta / v.sigma) ** 2
    return np.sum(np.asarray(heights) * np.exp(exponents))


def metadynamics_simulation(model, ufed):
    integrator = ufedmm.MiddleMassiveNHCIntegrator(TEMP, TAU, DT)
    simulation = reference_simulation(model, ufed, integrator)
    (metadynamics,) = simulation._periodic_tasks
    metadynamics.force.setForceGroup(METAD_GROUP)
    simulation.context.reinitialize(preserveState=True)
    recorder = HillRecorder(metadynamics)
    simulation.reporters.append(recorder)
    return simulation, recorder


def current_bias(simulation):
    state = simulation.context.getState(
        getPositions=True, getEnergy=True, groups={METAD_GROUP}
    )
    bias = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    return state.getDynamicalVariables(), bias


PERIODIC_1D = [("phi", 0.05), ("psi", None)]
PERIODIC_2D = [("phi", 0.05), ("psi", 0.05)]
NONPERIODIC_1D = [("distance", 0.05)]
MIXED_2D = [("phi", 0.05), ("distance", 0.05)]
PERIODIC_3D = [("phi", 0.1), ("psi", 0.1), ("omega", 0.1)]

VARIABLE_SETS = pytest.mark.parametrize(
    "specs",
    [PERIODIC_1D, PERIODIC_2D, NONPERIODIC_1D, MIXED_2D, PERIODIC_3D],
    ids=["periodic-1D", "periodic-2D", "nonperiodic-1D", "mixed-2D", "periodic-3D"],
)
GRIDS = pytest.mark.parametrize(
    "enforce_gridless, tolerance",
    [(False, 1e-4), (True, 1e-10)],
    ids=["gridded", "gridless"],
)


@GRIDS
@VARIABLE_SETS
def test_metadynamics_bias_is_a_sum_of_hills(specs, enforce_gridless, tolerance):
    model, ufed = metadynamics_model(specs, enforce_gridless=enforce_gridless)
    simulation, recorder = metadynamics_simulation(model, ufed)
    simulation.step(100)
    point, bias = current_bias(simulation)
    assert len(recorder.centers) == 10
    assert recorder.heights == pytest.approx([ufed.height] * 10)
    expected = hills_bias(ufed.variables, recorder.centers, recorder.heights, point)
    assert bias > 0
    assert bias == pytest.approx(expected, rel=tolerance)


def test_gridless_hills_buffer_grows_across_short_steps():
    # Steps shorter than the deposition period, and a buffer smaller than the number
    # of hills, exercise both ways the gridless force makes room for new hills.
    model, ufed = metadynamics_model(PERIODIC_2D, enforce_gridless=True)
    ufed.buffer_size = 2
    simulation, recorder = metadynamics_simulation(model, ufed)
    for _ in range(33):
        simulation.step(3)
    simulation.step(1)
    point, bias = current_bias(simulation)
    assert len(recorder.centers) == 10
    expected = hills_bias(ufed.variables, recorder.centers, recorder.heights, point)
    assert bias == pytest.approx(expected, rel=1e-10)


@GRIDS
@pytest.mark.parametrize(
    "specs", [PERIODIC_1D, PERIODIC_2D], ids=["periodic-1D", "periodic-2D"]
)
def test_well_tempered_metadynamics_scales_hill_heights(
    specs, enforce_gridless, tolerance
):
    bias_factor = 10
    model, ufed = metadynamics_model(
        specs, bias_factor=bias_factor, enforce_gridless=enforce_gridless
    )
    simulation, recorder = metadynamics_simulation(model, ufed)
    simulation.step(100)
    point, bias = current_bias(simulation)
    variables = ufed.variables
    delta_kT = (bias_factor - 1) * KB * variables[0].temperature
    expected_heights = [ufed.height]
    for k in range(1, len(recorder.centers)):
        previous = hills_bias(
            variables,
            recorder.centers[:k],
            recorder.heights[:k],
            recorder.centers[k],
        )
        expected_heights.append(ufed.height * np.exp(-previous / delta_kT))
    assert recorder.heights == pytest.approx(expected_heights, rel=tolerance)
    assert recorder.heights[-1] < 0.9 * ufed.height
    expected = hills_bias(variables, recorder.centers, recorder.heights, point)
    assert bias == pytest.approx(expected, rel=tolerance)


def test_well_tempered_metadynamics_requires_equal_temperatures():
    model, ufed = metadynamics_model(
        PERIODIC_2D, temperatures=[TS, 2 * TS], bias_factor=10
    )
    integrator = ufedmm.MiddleMassiveNHCIntegrator(TEMP, TAU, DT)
    with pytest.raises(ValueError, match="same temperature"):
        reference_simulation(model, ufed, integrator)
