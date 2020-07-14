"""
.. module:: cvlib
   :platform: Unix, Windows
   :synopsis: A collection of custom collective variables

.. moduleauthor:: Charlles Abreu <abreu@eq.ufrj.br>

.. _Context: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.Context.html
.. _CustomCVForce: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.CustomCVForce.html
.. _CustomIntegrator: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.CustomIntegrator.html
.. _Force: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.Force.html
.. _System: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.System.html

"""

from simtk import openmm, unit


class SquareRadiusOfGyration(openmm.CustomBondForce):
    """
    The square of the radius of gyration of a group of atoms, defined as:

    .. math::
        R_g^2 = \\frac{1}{n^2} \\sum_i \\sum_{j>i} r_{i,j}^2,

    where :math:`n` is the number of atoms in the group and :math:`r_{i,j}` is the distance between
    atoms `i` and `j`.

    Parameters
    ----------
        group : list(int)
            The indices of the atoms in the group.

    """

    def __init__(self, group):
        super().__init__(f'r^2/{len(group)**2}')
        self.setUsesPeriodicBoundaryConditions(False)
        for i in group[0:-2]:
            for j in group[i+1:]:
                self.addBond(i, j)


class CoordinationNumber(openmm.CustomNonbondedForce):
    """
     A continuos approximation for the number of neighbor pairs among atoms of two groups,
     defined as:

    .. math::
        N(\\mathbf{g}_1, \\mathbf{g}_2) = \\sum_{i \\in \\mathbf{g}_1} \\sum_{j \\in \\mathbf{g}_2}
                \\frac{S\\left(\\frac{r_{i,j}}{d_0}-1\\right)}{1+\\left(\\frac{r_{i,j}}{d_0}\\right)^n},

    where :math:`d_0` is a threshold distance, :math:`n` is a sharpness parameter, :math:`r_{ij}`
    is the distance between atoms :math:`i \\in \\mathbf{g}_1` and :math:`j \\in \\mathbf{g}_2`,
    and :math:`S(x)` is a switching function that acts for :math:`r_{i,j} \\geq d_0`, defined as

    .. math::
        S(x) = \\begin{cases}
                   1 & x < 0 \\\\
                   1-6x^5+15x^4-10x^3 & 0 \\leq x \\leq 1 \\\\
                   0 & x > 1
               \\end{cases}

    The function that is summed for each atom pair is a sigmoidal approximation for a step
    function that goes down from 1 to 0 at a distance :math:`d_0` and has the following shape:

    .. image::
        figures/coordination_number.png
        :align: center

    With :math:`n = 6` (default), this is the same function defined in :cite:`Iannuzzi_2003`,
    except that the function defined here decays more smoothly to zero throughout the interval
    :math:`r_{i,j} \\in [d_0, 2 d_0]`.

    .. warning::
        If the two specified atom groups share atoms, each pair `i,j` among these atoms will be
        counted only once.

    Parameters
    ----------
        system : openmm.System
            The system for which this collective variable will be computed.
        group1 : list(int)
            The indices of the atoms in the first group.
        group2 : list(int)
            The indices of the atoms in the second group.

    Keyword Args
    ------------
        d0 : unit.Quantity, default=4*unit.angstroms
            The threshold distance, which is also half the actual cutoff distance.
        m : int or float, default=6
            Exponent that controls the sharpness of the sigmoidal function.

    """
    def __init__(self, system, group1, group2, d0=4*unit.angstroms, n=6):
        super().__init__(f'1/(1+(r/d0)^{n})')
        if system.usesPeriodicBoundaryConditions():
            self.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
        else:
            self.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffNonPeriodic)
        for i in range(system.getNumParticles()):
            self.addParticle([])
        self.addGlobalParameter('d0', d0)
        self.setUseSwitchingFunction(True)
        self.setSwitchingDistance(d0)
        self.setCutoffDistance(2*d0)
        self.setUseLongRangeCorrection(False)
        self.addInteractionGroup(group1, group2)
