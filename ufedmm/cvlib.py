"""
.. module:: cvlib
   :platform: Unix, Windows
   :synopsis: A collection of custom collective variables

.. moduleauthor:: Charlles Abreu <abreu@eq.ufrj.br>

.. _Context: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.Context.html
.. _CustomCVForce: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.CustomCVForce.html
.. _CustomIntegrator: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.CustomIntegrator.html
.. _Force: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.Force.html
.. _NonbondedForce: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.NonbondedForce.html
.. _System: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.System.html

"""

import re
import itertools
import math

from collections import namedtuple
from simtk import openmm, unit
from ufedmm.ufedmm import _standardized


_ParamTuple = namedtuple('_ParamTuple', 'charge sigma epsilon')


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
        for i in group[0:-1]:
            for j in group[i+1:]:
                self.addBond(i, j)


class RadiusOfGyration(openmm.CustomCVForce):
    """
    The radius of gyration of a group of atoms, defined as:

    .. math::
        R_g = \\frac{1}{n} \\sqrt{\\sum_i \\sum_{j>i} r_{i,j}^2},

    where :math:`n` is the number of atoms in the group and :math:`r_{i,j}` is the distance between
    atoms `i` and `j`.

    Parameters
    ----------
        group : list(int)
            The indices of the atoms in the group.

    """

    def __init__(self, group):
        RgSq = openmm.CustomBondForce('r^2')
        RgSq.setUsesPeriodicBoundaryConditions(False)
        for i in group[0:-1]:
            for j in group[i+1:]:
                RgSq.addBond(i, j)
        super().__init__(f'sqrt(RgSq)/{len(group)}')
        self.addCollectiveVariable('RgSq', RgSq)


class CoordinationNumber(openmm.CustomNonbondedForce):
    """
     A continuos approximation for the number of neighbor pairs among atoms of two groups,
     defined as:

    .. math::
        N(\\mathbf{g}_1, \\mathbf{g}_2) = \\sum_{i \\in \\mathbf{g}_1} \\sum_{j \\in \\mathbf{g}_2}
                     S\\left(\\frac{d_{i,j}}{d_0}-1\\right) F_n \\left(\\frac{d_{i,j}}{d_0}\\right)

    where :math:`d_0` is a threshold distance and :math:`d_{ij}` is the distance between atoms
    :math:`i \\in \\mathbf{g}_1` and :math:`j \\in \\mathbf{g}_2`.
    The function :math:`F_n(x)` is a continuous step function defined as

    .. math::
        F_n(x) = \\frac{1}{1+x^n}

    where :math:`n` is a sharpness parameter. With :math:`n = 6` (default), this is the same
    function defined in :cite:`Iannuzzi_2003`. It has the following shape for varying `n` values:

    .. image::
        figures/coordination_number.png
        :align: center

    Besides, :math:`S(x)` is a switching function given by

    .. math::
        S(x) = \\begin{cases}
                   1 & x < 0 \\\\
                   1-6x^5+15x^4-10x^3 & 0 \\leq x \\leq 1 \\\\
                   0 & x > 1
               \\end{cases}

    Thus, the amount summed up for each atom pair decays smoothly to zero throughout the interval
    :math:`r_{i,j} \\in [d_0, 2 d_0]`, meaning that :math:`2 d_0` is an actual cutoff distance.

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


class HelixAngleContent(openmm.CustomAngleForce):
    """
     Fractional alpha-helix content of a sequence of residues in a protein chain based on the
     angles between consecutive alpha-carbon atoms, defined as follows:

    .. math::
        \\alpha_\\theta(r_M,\\cdots,r_N) = \\frac{1}{N-M-1} \\sum_{i=M+1}^{N-1} F_n\\left(
        \\frac{\\theta(\\mathrm{C}_\\alpha^{i-1},\\mathrm{C}_\\alpha^i,\\mathrm{C}_\\alpha^{i+1})
        - \\theta_\\mathrm{ref}}{\\theta_\\mathrm{tol}}\\right)

    where :math:`\\theta(\\mathrm{C}_\\alpha^{i-1},\\mathrm{C}_\\alpha^i,\\mathrm{C}_\\alpha^{i+1})`
    is the angle between three consecutive alpha-carbon atoms, :math:`\\theta_\\mathrm{ref}` is the
    reference value of this angle, and :math:`\\theta_\\mathrm{tol}` is the tolerance threshold
    around this reference.

    The function :math:`F_n(x)` is defined as in :class:`CoordinationNumber`, but only even integer
    values are accepted for `n`.

    Parameters
    ----------
        topology : openmm.app.Topology
            The topology of the system for which this collective variable will be computed.
        first, last : int
            The indices of the first and last residues involved in the alpha helix.

    Keyword Args
    ------------
        n : even integer, default=6
            Exponent that controls the sharpness of the sigmoidal function.
        theta_ref : unit.Quantity, default=88*unit.degrees
            The reference value of the alpha carbon angle in the alpha helix.
        theta_tol : unit.Quantity, default=*unit.degrees
            The tolerance for the deviation from the alpha carbon angle.

    """

    def __init__(self, topology, first, last,
                 n=6, theta_ref=88*unit.degrees, theta_tol=15*unit.degrees):
        residues = [r for r in topology.residues() if first <= r.index <= last]
        if len(set(r.chain.index for r in residues)) > 1:
            raise ValueError('AngleHelixContent requires all residues in a single chain')
        if n % 2 != 0:
            raise ValueError("AngleHelixContent requires n to be an even integer number")
        super().__init__(f'1/({last-first-1}*(1+x^{n})); x=(theta - theta_ref)/theta_tol')
        self.addGlobalParameter('theta_ref', theta_ref)
        self.addGlobalParameter('theta_tol', theta_tol)
        alpha_carbons = [atom.index for r in residues for atom in r.atoms() if atom.name == 'CA']
        for i, j, k in zip(alpha_carbons[0:-2], alpha_carbons[1:-1], alpha_carbons[2:]):
            self.addAngle(i, j, k, [])


class HelixHydrogenBondContent(openmm.CustomBondForce):
    """
     Fractional alpha-helix content of a sequence of residues in a protein chain based on the
     hydrogen bonds between oxygen atoms and H-N groups located four residues apart, defined as
     follows:

    .. math::
        \\alpha_\\mathrm{hb}(r_M,\\cdots,r_N) = \\frac{1}{M-N-2} \\sum_{i=M+2}^{N-2} F_n\\left(
        \\frac{d(\\mathrm{O}^{i-2}, \\mathrm{H}^{i+2})}{d_0}\\right)

    where :math:`d(\\mathrm{O}^{i-2}, \\mathrm{H}^{i+2})` is the distance between the oxygen and
    hydrogen atoms and :math:`d_0` is the threshold distance for characterizing a hydrogen bond.

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
        residues = [r for r in topology.residues() if first <= r.index <= last]
        if len(set(r.chain.index for r in residues)) > 1:
            raise ValueError('HelixHydrogenBondContent requires all residues in a single chain')
        super().__init__(f'1/({last-first-2}*(1+x^{n})); x=r/d0')
        self.addGlobalParameter('d0', d0)
        reH = re.compile('\\b(H|1H|HN1|HT1|H1|HN)\\b')
        reO = re.compile('\\b(O|OCT1|OC1|OT1|O1)\\b')
        oxygens = [atom.index for r in residues for atom in r.atoms() if re.match(reO, atom.name)]
        hydrogens = [atom.index for r in residues for atom in r.atoms() if re.match(reH, atom.name)]
        for i, j in zip(oxygens[:-3], hydrogens[3:]):
            self.addBond(i, j, [])


class HelixRamachandranContent(openmm.CustomTorsionForce):
    """
     Fractional alpha-helix content of a sequence of residues in a protein chain based on the
     Ramachandran dihedral angles, defined as follows:

    .. math::
        \\alpha_{\\phi,\\psi}(r_M,\\cdots,r_N) = \\frac{1}{2(N-M-1)} \\sum_{i=M+1}^{N-1} \\Bigg[
            F_n\\left(
                \\frac{\\phi(\\mathrm{C}^{i-1},\\mathrm{N}^i,\\mathrm{C}_\\alpha^i, \\mathrm{C}^i)
                - \\phi_\\mathrm{ref}}{\\phi_\\mathrm{tol}}
            \\right) + \\\\
            F_n\\left(
                \\frac{\\psi(\\mathrm{N}^i,\\mathrm{C}_\\alpha^i, \\mathrm{C}^i, \\mathrm{N}^{i+1})
                - \\psi_\\mathrm{ref}}{\\psi_\\mathrm{tol}}
            \\right)
        \\Bigg]

    where :math:`\\phi(\\mathrm{C}^{i-1},\\mathrm{N}^i,\\mathrm{C}_\\alpha^i, \\mathrm{C}^i)` and
    :math:`\\psi(\\mathrm{N}^i,\\mathrm{C}_\\alpha^i, \\mathrm{C}^i, \\mathrm{N}^{i+1})` are the
    Ramachandran dihedral angles, :math:`\\phi_\\mathrm{ref}` and :math:`\\psi_\\mathrm{ref}` are
    their reference values in an alpha helix, and :math:`\\phi_\\mathrm{tol}` and
    :math:`\\psi_\\mathrm{tol}` are the threshold tolerances around these refenrences.

    The function :math:`F_n(x)` is defined as in :class:`CoordinationNumber`, but only even integer
    values are accepted for `n`.

    Default values are the overall average alpha-helix dihedral angles and their dispersions
    reported in :cite:`Hovmoller_2002`.

    Parameters
    ----------
        topology : openmm.app.Topology
            The topology of the system for which this collective variable will be computed.
        first, last : int
            The indices of the first and last residues involved in the alpha helix.

    Keyword Args
    ------------
        n : even integer, default=6
            Exponent that controls the sharpness of the sigmoidal function.
        phi_ref : unit.Quantity, default=-63.8*unit.degrees
            The reference value of the Ramachandran :math:`\\phi` dihedral angle in the alpha helix.
        phi_tol : unit.Quantity, default=25*unit.degrees
            The tolerance for the deviation from the Ramachandran :math:`\\phi` dihedral angle.
        psi_ref : unit.Quantity, default=-41.1*unit.degrees
            The reference value of the Ramachandran :math:`\\psi` dihedral angle in the alpha helix.
        psi_tol : unit.Quantity, default=25*unit.degrees
            The tolerance for the deviation from the Ramachandran :math:`\\psi` dihedral angle.

    """

    def __init__(self, topology, first, last, n=6,
                 phi_ref=-63.8*unit.degrees, phi_tol=25*unit.degrees,
                 psi_ref=-41.1*unit.degrees, psi_tol=25*unit.degrees):

        residues = [r for r in topology.residues() if first <= r.index <= last]
        if len(set(r.chain.index for r in residues)) > 1:
            raise ValueError('HelixRamachandranContent requires all residues in a single chain')
        super().__init__(f'1/({2*(last-first)}*(1+x^{n})); x=(theta - theta_ref)/theta_tol')
        self.addPerTorsionParameter('theta_ref')
        self.addPerTorsionParameter('theta_tol')
        C = [atom.index for r in residues for atom in r.atoms() if atom.name == 'C']
        N = [atom.index for r in residues for atom in r.atoms() if atom.name == 'N']
        CA = [atom.index for r in residues for atom in r.atoms() if atom.name == 'CA']
        for i, j, k, l in zip(C[:-1], N[1:], CA[1:], C[1:]):
            self.addTorsion(i, j, k, l, [phi_ref, phi_tol])
        for i, j, k, l in zip(N[:-1], CA[:-1], C[:-1], N[1:]):
            self.addTorsion(i, j, k, l, [psi_ref, psi_tol])

    def atom_indices(self):
        """
        Returns
        -------
            phi_indices : list of tuples
                The indices of the atoms in the :math:`\\phi` dihedrals.
            psi_indices : list of tuples
                The indices of the atoms in the :math:`\\psi` dihedrals.

        """
        N = self.getNumTorsions()//2
        phi_indices = []
        psi_indices = []
        for index in range(N):
            i, j, k, l, parameters = self.getTorsionParameters(index)
            phi_indices.append((i, j, k, l))
            i, j, k, l, parameters = self.getTorsionParameters(index + N)
            psi_indices.append((i, j, k, l))
        return phi_indices, psi_indices


class _InOutForce(openmm.CustomNonbondedForce):
    """
    An abstract class for In/Out-force collective variables.

    """

    def _import_properties(self, nbforce):
        self.setNonbondedMethod(self.CutoffPeriodic)
        self.setCutoffDistance(nbforce.getCutoffDistance())
        self.setUseSwitchingFunction(nbforce.getUseSwitchingFunction())
        self.setSwitchingDistance(nbforce.getSwitchingDistance())

    def _update_nonbonded_force(self, group, nbforce, parameters, pbc_for_exceptions):
        internal_exception_pairs = []
        for index in range(nbforce.getNumExceptions()):
            i, j, _, _, epsilon = nbforce.getExceptionParameters(index)
            i_in_group, j_in_group = i in group, j in group
            if i_in_group and j_in_group:
                internal_exception_pairs.append(set([i, j]))
            elif (i_in_group or j_in_group):
                raise ValueError("No exceptions are allowed for in-group/out-group interactions")

        for i, j in itertools.combinations(group, 2):
            if set([i, j]) not in internal_exception_pairs:
                chargeprod = parameters[i].charge*parameters[j].charge
                sigma = (parameters[i].sigma + parameters[j].sigma)/2
                epsilon = unit.sqrt(parameters[i].epsilon*parameters[j].epsilon)
                nbforce.addException(i, j, chargeprod, sigma, epsilon)

        if pbc_for_exceptions:
            nbforce.setExceptionsUsePeriodicBoundaryConditions(True)


class InOutLennardJonesForce(_InOutForce):
    """
    Lennard-Jones (LJ) interactions between the atoms of a specified group and all other atoms
    in the system, referred to as in/out LJ interactions. All LJ parameters are imported from a
    provided NonbondedForce_ object, which is then modified so that all in-group interactions are
    treated as exceptions and all atoms of the group are removed from regular LJ interactions.

    .. note::
        No exceptions which involve in/out atom pairs are allowed.

    Warnings
    --------
        side effect:
            The constructor of this class modifies the passed NonbondedForce_ object.

    The model equation is:

    .. math::
        V_\\mathrm{LJ}(\\mathbf{r}) = \\sum_{i \\in \\mathcal{G}} \\sum_{j \\notin \\mathcal{G}}
        \\epsilon_{ij} u_\\mathrm{LJ}\\left(\\frac{r_{ij}}{\\sigma_{ij}}\\right)

    where :math:`\\mathcal{G}` is the specified group and

    .. math::
        u_\\mathrm{LJ}(x) = 4(x^{-12} - x^{-6})

    Parameters
    ----------
        group : list of int
            The atoms in the specified group.
        nbforce : openmm.NonbondedForce
            The NonbondedForce_ object from which the atom parameters are imported.

    Keyword Args
    ------------
        pbc_for_exceptions : bool, default=False
            Whether to consider periodic boundary conditions for exceptions in the NonbondedForce_
            object. This might be necessary if the specified group contains several detached
            molecules or one long molecule.

    Raises
    ------
        ValueError:
            Raised if there are any exceptions in the NonbondedForce_ object involving
            cross-group (i.e. in/out) atom pairs.

    """

    def __init__(self, group, nbforce, pbc_for_exceptions=False):
        u_LJ = '4/x^12-4/x^6'
        definitions = ['x=r/sigma', 'sigma=(sigma1+sigma2)/2', 'epsilon=sqrt(epsilon1*epsilon2)']
        equations = [f'epsilon*({u_LJ})'] + definitions
        super().__init__(';'.join(equations))
        N = nbforce.getNumParticles()
        parameters = [_ParamTuple(*nbforce.getParticleParameters(i)) for i in range(N)]
        self.addPerParticleParameter('sigma')
        self.addPerParticleParameter('epsilon')
        for parameter in parameters:
            self.addParticle([parameter.sigma, parameter.epsilon])
        self._update_nonbonded_force(group, nbforce, parameters, pbc_for_exceptions)
        self._import_properties(nbforce)
        self.addInteractionGroup(set(group), set(range(N)) - set(group))
        self.setUseLongRangeCorrection(nbforce.getUseDispersionCorrection())
        for i in group:
            nbforce.setParticleParameters(i, parameters[i].charge, 1.0, 0.0)

    def capped_version(self, m=2):
        """
        Returns a capped (Buelens-Grubmüller-type) version of the in/out Lennard-Jones force.

        The model equation is

        .. math::
            V_\\mathrm{BG}(\\mathbf{r}) = \\sum_{i \\in \\mathcal{G}} \\sum_{j \\notin \\mathcal{G}}
                \\epsilon_{ij} u_\\mathrm{BG}\\left(\\frac{r_{ij}}{\\sigma_{ij}}\\right)

        where

        .. math::
            u_\\mathrm{BG}(x) = \\left\\{ \\begin{array}{ccc}
                u_\\mathrm{cap}(x) & \\mathrm{if} & x < 1 \\\\
                4(x^{-12} - x^{-6}) & \\mathrm{if} & x \\geq 1
            \\end{array}\\right.

        with

        .. math::
            u_\\mathrm{cap}(x) = \\left\\{ \\begin{array}{ccc}
                126 x^4 - 176 x^3 + 50 & \\mathrm{if} & m = 2 \\\\
                \\frac{-4340 x^6 + 10944 x^5 - 7200 x^4 + 596}{5} & \\mathrm{if} & m = 3 \\\\
                \\frac{43365 x^8 - 155880 x^7 + 191065 x^6 - 80472 x^5 + 1922}{35} & \\mathrm{if} & m = 4
            \\end{array}\\right.

        Keyword Args
        ------------
            m : int, default=2
                The highest order of derivatives to be zero at :math:`r=0` and to match the
                Lennard-Jones values at :math:`r=\\sigma`. Valid options are 2, 3, and 4.

        """

        u_LJ = '4/x^12-4/x^6'
        if m == 2:
            u_cap = '126*x^4-176*x^3+50'
        elif m == 3:
            u_cap = '(-4340*x^6+10944*x^5-7200*x^4+596)/5'
        elif m == 4:
            u_cap = '(43365*x^8-155880*x^7+191065*x^6-80472*x^5+1922)/35'
        else:
            raise ValueError("Raised if an invalid `m` keyword value is passed.")
        definitions = ['x=r/sigma', 'sigma=(sigma1+sigma2)/2', 'epsilon=sqrt(epsilon1*epsilon2)']
        equations = [f'epsilon*select(step(1-x),{u_cap},{u_LJ})'] + definitions
        force = openmm.CustomNonbondedForce(';'.join(equations))
        force.addPerParticleParameter('sigma')
        force.addPerParticleParameter('epsilon')
        for index in range(self.getNumParticles()):
            force.addParticle(self.getParticleParameters(index))
        for index in range(self.getNumExclusions()):
            force.addExclusion(*self.getExclusionParticles(index))
        force.setNonbondedMethod(force.CutoffPeriodic)
        force.setCutoffDistance(self.getCutoffDistance())
        force.setUseSwitchingFunction(self.getUseSwitchingFunction())
        force.setSwitchingDistance(self.getSwitchingDistance())
        force.setUseLongRangeCorrection(self.getUseLongRangeCorrection())
        force.addInteractionGroup(*self.getInteractionGroupParameters(0))
        return force


class InOutCoulombForce(_InOutForce):
    """
    Cut-off, pairwise Coulomb interactions between the atoms of a specified group and al other atoms
    in the system, referred to as in/out Coulomb interactions. All charges are imported from a
    provided NonbondedForce_ object, which is then modified so that all in-group interactions are
    treated as exceptions and all charges of the group atoms are scaled by a newly created Context_
    global parameter whose default value is 0.0.

    .. note::
        No exceptions which involve in/out atom pairs are allowed.

    Warnings
    --------
        side effect:
            The constructor of this class modifies the passed NonbondedForce_ object.

    The model equation is

    .. math::
        V_\\mathrm{coul}(\\mathbf{r}) = \\sum_{i \\in \\mathcal{G}} \\sum_{j \\notin \\mathcal{G}}
            \\frac{q_i q_j}{4 \\pi \\epsilon_0 r_c} u\\left(\\frac{r}{r_c}\\right)

    where :math:`\\mathcal{G}` is the specified group and the function :math:`u(x)` can be
    chosen from a number of different styles:

    1. Shifted:

    .. math::
        u(x) = \\frac{1}{x} - 1

    2. Shifted-force:

    .. math::
        u(x) = \\frac{1}{x} + x - 2

    3. Conductor Reaction-field (default):

        .. math::
            u(x) = \\frac{1}{x} + \\frac{x^2}{2} - \\frac{3}{2}

    4. Reaction-field (with finite dielectric constant :math:`\\epsilon`):

        .. math::
            u(x) = \\frac{1}{x} + \\frac{(2\\epsilon-1)x^2-3\\epsilon}{2\\epsilon+1}

    5. Damped:

    .. math::
        u(x) = \\frac{\\mathrm{erfc}(\\alpha_c x)}{x}

    6. Damped-shifted-force (DSF), with :math:`\\alpha_c = \\alpha r_c`:

    .. math::
        u(x) = \\frac{\\mathrm{erfc}(\\alpha_c x)}{x} - \\mathrm{erfc}(\\alpha_c) +
        \\left[\\mathrm{erfc}(\\alpha_c) +
        \\frac{2\\alpha_c e^{-\\alpha_c^2}}{\\sqrt{\\pi}}\\right]\\left(x - 1\\right)

    Parameters
    ----------
        group : list of int
            The atoms in the specified group.
        nbforce : openmm.NonbondedForce
            The NonbondedForce_ object from which the atom charges are imported.

    Keyword Args
    ------------
        style : str, default='conductor-reaction-field'
            The style of cutoff electrostatic potential to be used. Valid options are `shifted`,
            `shifted-force`, `reaction-field`, `conductor-reaction-field`, `damped`, and
            `damped-shifted-force`.
        damping_coefficient : float or unit.Quantity, default=0.2/unit.angstroms
            The damping coefficient :math:`\\alpha` in inverse distance unit.
        scaling_parameter_name : str, default='inOutCoulombScaling'
            A Context_ global parameter whose value will multiply, in the passed NonbondedForce_
            object, the epsilon parameters of all atoms in the specified group.
        pbc_for_exceptions : bool, default=False
            Whether to consider periodic boundary conditions for exceptions in the NonbondedForce_
            object. This might be necessary if the specified group contains several detached
            molecules or one long molecule.

    Raises
    ------
        ValueError:
            Raised if there are any exceptions in the NonbondedForce_ object involving
            cross-group (i.e. in/out) atom pairs.
        ValueError:
            Raised if an invalid `style` keyword value is passed.

    """

    def __init__(self, group, nbforce, style='conductor-reaction-field',
                 damping_coefficient=0.2/unit.angstroms,
                 scaling_parameter_name='inOutCoulombScaling', pbc_for_exceptions=False):

        rc = _standardized(nbforce.getCutoffDistance())
        alpha_c = _standardized(damping_coefficient)*rc
        prefix = f'{138.935485/rc}*charge1*charge2'
        if style == 'shifted':
            u_C = '1/x - 1'
        elif style == 'shifted-force':
            u_C = '1/x + x - 2'
        elif style == 'conductor-reaction-field':
            u_C = '1/x + x^2/2 - 3/2'
        elif style == 'reaction-field':
            epsilon = nbforce.getReactionFieldDielectric()
            krf = (epsilon - 1)/(2*epsilon + 1)
            crf = 3*epsilon/(2*epsilon + 1)
            u_C = f'1/x + {krf}*x^2 - {crf}'
        elif style == 'damped':
            u_C = f'erfc({alpha_c}*x)/x'
        elif style == 'damped-shifted-force':
            A = math.erfc(alpha_c)
            B = A + 2*alpha_c*math.exp(-alpha_c**2)/math.sqrt(math.pi)
            u_C = f'erfc({alpha_c}*x)/x + {A+B}*x - {2*A+B}'
        else:
            raise ValueError("Invalid cutoff electrostatic style")

        super().__init__(f'{prefix}*({u_C}); x=r/{rc}')
        N = nbforce.getNumParticles()
        parameters = [_ParamTuple(*nbforce.getParticleParameters(i)) for i in range(N)]
        offset_index = {}
        for index in range(nbforce.getNumParticleParameterOffsets()):
            variable, i, charge, _, _ = nbforce.getParticleParameterOffset(index)
            if variable == scaling_parameter_name:
                offset_index[i] = index
                parameters[i] = _ParamTuple(charge, parameters[i].sigma, parameters[i].epsilon)
        self.addPerParticleParameter('charge')
        for parameter in parameters:
            self.addParticle([parameter.charge])
        self._update_nonbonded_force(group, nbforce, parameters, pbc_for_exceptions)
        self._import_properties(nbforce)
        self.addInteractionGroup(set(group), set(range(N)) - set(group))
        self.setUseLongRangeCorrection(False)
        global_vars = map(nbforce.getGlobalParameterName, range(nbforce.getNumGlobalParameters()))
        if scaling_parameter_name not in global_vars:
            nbforce.addGlobalParameter(scaling_parameter_name, 0.0)
        for i in group:
            charge, sigma, epsilon = parameters[i]
            nbforce.setParticleParameters(i, 0.0, sigma, epsilon)
            input = [scaling_parameter_name, i, charge, 0.0, 0.0]
            if i in offset_index:
                nbforce.setParticleParameterOffset(offset_index[i], *input)
            else:
                nbforce.addParticleParameterOffset(*input)
