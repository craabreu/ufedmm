"""
Unit and regression test for the ufedmm package.
"""

# Import package, test suite, and other packages as needed
import ufedmm
# import pytest
import sys
import yaml

from simtk import unit


def ufed_model():
    model = ufedmm.AlanineDipeptideModel()
    mass = 50*unit.dalton*(unit.nanometer/unit.radians)**2
    Ks = 1000*unit.kilojoules_per_mole/unit.radians**2
    Ts = 1500*unit.kelvin
    limit = 180*unit.degrees
    sigma = 18*unit.degrees
    height = 2.0*unit.kilojoules_per_mole
    frequency = 10
    s_phi = ufedmm.DynamicalVariable('s_phi', -limit, limit, mass, Ts, model.phi, Ks, sigma)
    s_psi = ufedmm.DynamicalVariable('s_psi', -limit, limit, mass, Ts, model.psi, Ks, sigma)
    return model, ufedmm.UnifiedFreeEnergyDynamics([s_phi, s_psi], 300*unit.kelvin, height, frequency)


def test_ufedmm_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "ufedmm" in sys.modules


def test_serialization():
    model, old = ufed_model()
    new = yaml.load(yaml.dump(old), Loader=yaml.FullLoader)
    assert new.__repr__() == old.__repr__()
    for var1, var2 in zip(old.variables, new.variables):
        assert var1.__repr__() == var2.__repr__()
