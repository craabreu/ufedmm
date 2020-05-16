"""
.. module:: testmodels
   :platform: Unix, Windows
   :synopsis: Unified Free Energy Dynamics Test Model Systems

.. moduleauthor:: Charlles Abreu <abreu@eq.ufrj.br>

.. _System: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.System.html
.. _Topology: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.app.topology.Topology.html

"""

import ufedmm
import os

from simtk import openmm, unit
from simtk.openmm import app


class AlanineDipeptideModel(object):
    """
    A system consisting of a single alanine-dipeptide molecule in a vacuum or solvated in explicit
    water.

    Keyword Args
    ------------
        force_field : str, default="amber03"
            The force field to be used for the alanine dipeptide molecule.
        water : str, default=None
            The water model to be used if the alanine dipeptide is supposed to be solvated.
            Available options are "spce", "tip3p", "tip4pew", and "tip5p".
        box_length : unit.Quantity, default=25*unit.angstroms
            The size of the simulation box. This is only effective if water is not `None`.

    Properties
    ----------
        system : openmm.System
            The system.
        topology : openmm.app.Topology
            The topology.
        positions : list of openmm.Vec3
            The positions.
        phi : openm.CustomTorsionForce
            The Ramachandran dihedral angle :math:`\\phi` of the alanine dipeptide molecule.
        psi : openm.CustomTorsionForce
            The Ramachandran dihedral angle :math:`\\psi` of the alanine dipeptide molecule.

    """

    def __init__(self, force_field='amber03', water=None, box_length=25*unit.angstroms):
        pdb = app.PDBFile(os.path.join(ufedmm.__path__[0], 'data', 'alanine-dipeptide.pdb'))
        if water is None:
            force_field = app.ForceField(f'{force_field}.xml')
            self.topology = pdb.topology
            self.positions = pdb.positions
            L = box_length.value_in_unit(unit.nanometers)
            vectors = [openmm.Vec3(L, 0, 0), openmm.Vec3(0, L, 0), openmm.Vec3(0, 0, L)]
            self.topology.setPeriodicBoxVectors(vectors)
        else:
            force_field = app.ForceField(f'{force_field}.xml', f'{water}.xml')
            modeller = app.Modeller(pdb.topology, pdb.positions)
            modeller.addSolvent(force_field, model=water, boxSize=box_length*openmm.Vec3(1, 1, 1))
            self.topology = modeller.topology
            self.positions = modeller.positions
        self.system = force_field.createSystem(
            self.topology,
            nonbondedMethod=app.NoCutoff if water is None else app.PME,
            constraints=app.HBonds,
            rigidWater=True,
            removeCMMotion=False,
        )
        atoms = [(a.name, a.residue.name) for a in self.topology.atoms()]
        phi_atoms = [('C', 'ACE'), ('N', 'ALA'), ('CA', 'ALA'), ('C', 'ALA')]
        self.phi = ufedmm.CollectiveVariable('phi', openmm.CustomTorsionForce('theta'))
        self.phi.force.addTorsion(*[atoms.index(i) for i in phi_atoms], [])
        psi_atoms = [('N', 'ALA'), ('CA', 'ALA'), ('C', 'ALA'), ('N', 'NME')]
        self.psi = ufedmm.CollectiveVariable('psi', openmm.CustomTorsionForce('theta'))
        self.psi.force.addTorsion(*[atoms.index(i) for i in psi_atoms], [])
