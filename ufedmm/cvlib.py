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

from simtk import openmm


class SquareRadiusOfGyration(openmm.CustomBondForce):
    """
    The square of the radius of gyration of a group of atoms, defined as:

    .. math::
        r_g^2 = \\frac{1}{N^2} \\sum_i \\sum_{j>i} \\|\\mathbf{r}_j - \\mathbf{r}_j\\|^2

    Parameters
    ----------
        group : list(int)
            The indices of the atoms in the group.

    """

    def __init__(self, group):
        super().__init__(f'r^2/{len(group)**2}')
        for i in group[0:-2]:
            for j in group[i+1:]:
                self.addBond(i, j)
