import re

import ufedmm


def test_version_is_exported():
    assert re.fullmatch(r"\d+\.\d+\.\d+.*", ufedmm.__version__)


def test_cvpack_reexport():
    import cvpack

    import ufedmm.cvpack

    assert ufedmm.cvpack.Torsion is cvpack.Torsion
