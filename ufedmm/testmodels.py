"""
.. module:: testmodels
   :platform: Unix, Windows
   :synopsis: Adiabatic Free Energy Dynamics Test Model Systems

.. moduleauthor:: Charlles Abreu <abreu@eq.ufrj.br>

.. _System: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.System.html
.. _Topology: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.app.topology.Topology.html

"""

import ufedmm
import os

from copy import deepcopy
from simtk import openmm, unit
from simtk.openmm import app


class TestModel:
    def createDefaultIntegrator(self, conservedEnergy=False):
        """
        Creates an integrator object corresponding to a massive Nosé-Hoover chain algorithm with
        :math:`T = 300~\\mathrm{K}`, :math:`\\tau = 100~\\mathrm{fs}`, and
        :math:`\\Delta t = 1~\\mathrm{fs}`.

        Returns
        -------
            MassiveMiddleNHCIntegrator

        """
        integrator = ufedmm.MassiveMiddleNHCIntegrator(
            300*unit.kelvin,
            100*unit.femtoseconds,
            1*unit.femtosecond,
            self.getDrivingForce(),
            conservedEnergy=conservedEnergy,
        )
        return integrator

    def createDefaultSimulation(self, platform='Reference', properties={}, conservedEnergy=False):
        """
        Creates a simulation object corresponding to a massive Nosé-Hoover chain algorithm with
        :math:`T = 300~\\mathrm{K}`, :math:`\\tau = 100~\\mathrm{fs}`, and
        :math:`\\Delta t = 1~\\mathrm{fs}`.

        Keyword Args
        ------------
            platform : str, default='Reference'
                The name of the OpenMM platform to be used for the computations.
            properties : dict, default={}
                The properties of the OpenMM platform to be used for the computations.

        Returns
        -------
            openmm.Simulation

        """
        platform = openmm.Platform.getPlatformByName(platform)
        simulation = openmm.app.Simulation(
            self.getTopology(),
            self.getSystem(),
            self.createDefaultIntegrator(conservedEnergy=conservedEnergy),
            platform,
            properties,
        )
        simulation.context.setPositions(self.getPositions())
        return simulation

    def getDrivingForce(self):
        """
        Gets the driving force associated to the model system.

        Returns
        -------
            DrivingForce

        """
        return self._driving_force

    def getPositions(self):
        """
        Gets the positions of all atoms in the model system.

        Returns
        -------
            list(openmm.Vec3)

        """
        return self._positions

    def getSystem(self):
        """
        Gets the System_ object of this model system.

        Returns
        -------
            openmm.System

        """
        return self._system

    def getTopology(self):
        """
        Gets the Topology_ object of this model system.

        Returns
        -------
            openmm.app.Topology

        """
        return self._topology


class AlanineDipeptideModel(TestModel):
    """
    A system consisting of a single alanine-dipeptide molecule in a vacuum or solvated in explicit
    water.

    Keyword Args
    ------------
        forceField : str, default="amber14-all"
            The force field to be used for the alanine dipeptide molecule.
        water : str, default=None
            The water model to be used if the alanine dipeptide is supposed to be solvated.
            Available options are "spce", "tip3p", "tip4pew", and "tip5p".
        boxLength : unit.Quantity, default=25*unit.angstroms
            The size of the simulation box. This is only effective if water is not `None`.
        bareSystem : bool, default=False
            Whether the system must be created without AFED-related objects.

    """
    def __init__(self, forceField='amber14-all', water=None, boxLength=25*unit.angstroms,
                 bareSystem=False):
        pdb = app.PDBFile(os.path.join(ufedmm.__path__[0], 'data', 'alanine-dipeptide.pdb'))
        if water is None:
            force_field = app.ForceField(f'{forceField}.xml')
            self._topology = pdb.topology
            self._positions = pdb.positions
        else:
            force_field = app.ForceField(f'{forceField}.xml', f'{water}.xml')
            modeller = app.Modeller(pdb.topology, pdb.positions)
            modeller.addSolvent(force_field, model=water, boxSize=boxLength*openmm.Vec3(1, 1, 1))
            self._topology = modeller.topology
            self._positions = modeller.positions
        self._system = force_field.createSystem(
            self._topology,
            nonbondedMethod=app.NoCutoff if water is None else app.PME,
            constraints=None,
            rigidWater=False,
            removeCMMotion=False,
        )
        if bareSystem:
            return
        atoms = [(a.name, a.residue.name) for a in self._topology.atoms()]
        psi_atoms = [('N', 'ALA'), ('CA', 'ALA'), ('C', 'ALA'), ('N', 'NME')]
        self._psi_angle = openmm.CustomTorsionForce('theta')
        self._psi_angle.addTorsion(*[atoms.index(i) for i in psi_atoms], [])
        phi_atoms = [('C', 'ACE'), ('N', 'ALA'), ('CA', 'ALA'), ('C', 'ALA')]
        self._phi_angle = openmm.CustomTorsionForce('theta')
        self._phi_angle.addTorsion(*[atoms.index(i) for i in phi_atoms], [])
        period = 360*unit.degrees
        self._psi = ufedmm.DrivenCollectiveVariable('psi', self._psi_angle, unit.radians, period)
        self._phi = ufedmm.DrivenCollectiveVariable('phi', self._phi_angle, unit.radians, period)
        value = 180*unit.degrees
        T = 1500*unit.kelvin
        mass = 168.0*unit.dalton*(unit.angstroms/unit.radian)**2
        velocity_scale = unit.sqrt(unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA*T/mass)
        self._psi_driver = ufedmm.DriverParameter('psi_s', unit.radians, value, T, velocity_scale,
                                                period=period)
        self._phi_driver = ufedmm.DriverParameter('phi_s', unit.radians, value, T, velocity_scale,
                                                period=period)
        self._driving_force = ufedmm.HarmonicDrivingForce()
        K = 2.78E3*unit.kilocalories_per_mole/unit.radians**2
        self._driving_force.addPair(self._psi, self._psi_driver, K)
        self._driving_force.addPair(self._phi, self._phi_driver, K)
        self._system.addForce(self._driving_force)

    def getDihedralAngles(self):
        """
        Gets the Ramachandran dihedral angles :math:`\\psi` and :math:`\\phi` of the alanine
        dipeptide angles.

        Returns
        -------
            psi_angle, phi_angle : openmm.CustomTorsionForce

        """
        return self._psi_angle, self._phi_angle

    def getCollectiveVariables(self, copy=False):
        """
        Gets driven collective variables concerning the Ramachandran dihedral angles.

        .. warning::
            The returned :class:`~ufedmm.ufedmm.DrivenCollectiveVariable` objects will be bound to an existing
            :class:`~ufedmm.ufedmm.HarmonicDrivingForce`, unless the keyword ``copy`` is set to ``True``.

        Keyword Args
        ------------
            copy : bool, default=False
                Whether to return a copy of the driven collective variables instead of the
                original objects.

        Returns
        -------
            psi, phi : DrivenCollectiveVariable

        """
        psi = deepcopy(self._psi) if copy else self._psi
        phi = deepcopy(self._phi) if copy else self._phi
        return psi, phi

    def getDriverParameters(self):
        """
        Gets the driver parameters associated to the Ramachandran dihedral angles.

        Returns
        -------
            psi_driver, phi_driver : DrivenCollectiveVariable

        """
        return self._psi_driver, self._phi_driver
