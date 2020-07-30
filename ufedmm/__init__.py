"""
UFED for OpenMM
Unified Free Energy Dynamics with OpenMM

"""


from ._version import get_versions
from .ufedmm import CollectiveVariable  # noqa: F401, F403
from .ufedmm import DynamicalVariable  # noqa: F401, F403
from .ufedmm import ExtendedSpaceSimulation  # noqa: F401, F403
from .ufedmm import UnifiedFreeEnergyDynamics  # noqa: F401, F403
from .integrators import *  # noqa: F401, F403
from .io import *  # noqa: F401, F403
from .analysis import *  # noqa: F401, F403
from .testmodels import *  # noqa: F401, F403

# Handle versioneer:
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
