import re

import cvpack

import ufedmm
import ufedmm.cvpack


def test_version_is_exported():
    assert re.fullmatch(r"\d+\.\d+\.\d+.*", ufedmm.__version__)


def test_cvpack_reexport():
    assert ufedmm.cvpack.Torsion is cvpack.Torsion
