import openmm
from openmm import unit

import ufedmm


def test_no_spurious_version_errors_on_future_openmm(monkeypatch):
    # Version checks compared strings, so "10.0" < "7.5" was True.
    monkeypatch.setattr(openmm, "__version__", "10.0")
    model = ufedmm.AlanineDipeptideModel()
    ufedmm.add_inner_nonbonded_force(
        model.system, 5 * unit.angstroms, 8 * unit.angstroms, 1
    )
    ufedmm.MiddleMassiveNHCIntegrator(
        300 * unit.kelvin,
        10 * unit.femtoseconds,
        1 * unit.femtoseconds,
        respa_loops=[2, 1],
        embodied_force_groups=[1],
    )
