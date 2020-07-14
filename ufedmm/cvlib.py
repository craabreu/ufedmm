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
                                          F_n \\left(\\frac{d_{i,j}}{d_0}\\right)

    where :math:`d_0` is a threshold distance and :math:`d_{ij}` is the distance between atoms
    :math:`i \\in \\mathbf{g}_1` and :math:`j \\in \\mathbf{g}_2`.
    The function :math:`F_n(x)` is a continuous step function defined as

    .. math::
        F_n(x) = \\frac{S(x-1)}{1+x^n}

    where :math:`n` is a sharpness parameter and :math:`S(x)` is a switching function given by

    .. math::
        S(x) = \\begin{cases}
                   1 & x < 0 \\\\
                   1-6x^5+15x^4-10x^3 & 0 \\leq x \\leq 1 \\\\
                   0 & x > 1
               \\end{cases}

    It has the following shape:

    .. image::
        figures/coordination_number.png
        :align: center

    With :math:`n = 6` (default), the amount summed up for each atom pair is the same as the one
    defined in :cite:`Iannuzzi_2003`, except that here it decays more smoothly to zero throughout
    the interval :math:`r_{i,j} \\in [d_0, 2 d_0]`, meaning that :math:`2 d_0` is an actual cutoff
    distance.

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
        n : int or float, default=6
            Exponent that controls the sharpness of the sigmoidal function.
        d0 : unit.Quantity, default=4*unit.angstroms
            The threshold distance, which is also half the actual cutoff distance.

    """

    def __init__(self, system, group1, group2, n=6, d0=4*unit.angstroms):
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


class AngleHelixContent(openmm.CustomCompoundBondForce):
    """
     Fractional alpha-helix content of a sequence of residues in a protein chain based on the
     angles between consecutive alpha-carbon atoms, defined as follows:

    .. math::
        \\alpha_\\theta(r_M,\\cdots,r_N) = \\frac{1}{N-M-1} \\sum_{i=M+1}^{N-1} F_n\\left(
        \\frac{|\\theta(\\mathrm{C}_\\alpha^{i-1},\\mathrm{C}_\\alpha^i,\\mathrm{C}_\\alpha^{i+1})
        - \\theta_\\mathrm{ref}|}{\\delta \\theta_0}\\right)

    where :math:`\\theta(\\mathrm{C}_\\alpha^i,\\mathrm{C}_\\alpha^{i+1},\\mathrm{C}_\\alpha^{i+2})`
    is the angle between three consecutive alpha-carbon atoms.

    The function :math:`F_n(x)` is defined as in :class:`CoordinationNumber`.

    Parameters
    ----------
        topology : openmm.app.Topology
            The topology of the system for which this collective variable will be computed.
        first, last : int
            The indices of the first and last residues involved in the alpha helix.

    Keyword Args
    ------------
        n : int or float, default=6
            Exponent that controls the sharpness of the sigmoidal function.
        theta_ref : unit.Quantity, default=88*unit.degrees
            The reference value of the alpha carbon angle in the alpha helix.
        theta_tol : unit.Quantity, default=*unit.degrees
            The tolerance for the deviation from the alpha carbon angle.

    """

    def __init__(self, topology, first, last,
                 n=6, theta_ref=88*unit.degrees, theta_tol=15*unit.degrees):
        pass


class HydrogenBondHelixContent(openmm.CustomCompoundBondForce):
    """
     Fractional alpha-helix content of a sequence of residues in a protein chain based on the
     hydrogen bonds between oxygen atoms and H-N groups located four residues apart, defined as
     follows:

    .. math::
        \\alpha_\\mathrm{hb}(r_M,\\cdots,r_N) = \\frac{1}{M-N-3} \\sum_{i=M+2}^{N-2} F_n\\left(
        \\frac{d(\\mathrm{O}^{i-2}, \\mathrm{H}^{i+2})}{d_0}\\right)

    where :math:`d(\\mathrm{O}^{i-2}, \\mathrm{H}^{i+2})` is the distance between the oxygen and
    hydrogen atoms.

    The function :math:`F_n(x)` is defined as in :class:`CoordinationNumber`.

    Parameters
    ----------
        topology : openmm.app.Topology
            The topology of the system for which this collective variable will be computed.
        first, last : int
            The indices of the first and last residues involved in the alpha helix.

    Keyword Args
    ------------
        n : int or float, default=6
            Exponent that controls the sharpness of the sigmoidal function.
        d0 : unit.Quantity, default=4*unit.angstroms
            The threshold distance, which is also half the actual cutoff distance.

    """

    def __init__(self, topology, first, last, n=6, d0=3.3*unit.angstroms):
        pass


class RamachandranHelixContent(openmm.CustomCompoundBondForce):
    """
     Fractional alpha-helix content of a sequence of residues in a protein chain based on the
     Ramachandran dihedral angles, defined as follows:

    .. math::
        \\alpha_{\\phi,\\psi}(r_M,\\cdots,r_N) = \\frac{1}{2(N-M)} \\sum_{i=M+1}^N \\Bigg[
            F_n\\left(
                \\frac{|\\phi(\\mathrm{C}^{i-1},\\mathrm{N}^i,\\mathrm{C}_\\alpha^i, \\mathrm{C}^i)
                - \\phi_\\mathrm{ref}|}{\\delta \\phi_0}
            \\right) + \\\\
            F_n\\left(
                \\frac{|\\psi(\\mathrm{N}^i,\\mathrm{C}_\\alpha^i, \\mathrm{C}^i, \\mathrm{N}^{i+1})
                - \\psi_\\mathrm{ref}|}{\\delta \\psi_0}
            \\right)
        \\Bigg]

    where :math:`\\phi(\\mathrm{C}^{i-1},\\mathrm{N}^i,\\mathrm{C}_\\alpha^i, \\mathrm{C}^i)` and
    :math:`\\psi(\\mathrm{N}^i,\\mathrm{C}_\\alpha^i, \\mathrm{C}^i, \\mathrm{N}^{i+1})` are the
    Ramachandran dihedral angles.

    The function :math:`F_n(x)` is defined as in :class:`CoordinationNumber`.

    Parameters
    ----------
        topology : openmm.app.Topology
            The topology of the system for which this collective variable will be computed.
        first, last : int
            The indices of the first and last residues involved in the alpha helix.

    Keyword Args
    ------------
        n : int or float, default=6
            Exponent that controls the sharpness of the sigmoidal function.
        phi_ref : unit.Quantity, default=*unit.degrees
            The reference value of the Ramachandran :math:`\\phi` dihedral angle in the alpha helix.
        phi_tol : unit.Quantity, default=*unit.degrees
            The tolerance for the deviation from the Ramachandran :math:`\\phi` dihedral angle.
        psi_ref : unit.Quantity, default=*unit.degrees
            The reference value of the Ramachandran :math:`\\psi` dihedral angle in the alpha helix.
        psi_tol : unit.Quantity, default=*unit.degrees
            The tolerance for the deviation from the Ramachandran :math:`\\psi` dihedral angle.

    """

    def __init__(self, topology, first, last, n=6,
                 phi_ref=-60*unit.degrees, phi_tol=15*unit.degrees,
                 psi_ref=-60*unit.degrees, psi_tol=15*unit.degrees):
        pass
