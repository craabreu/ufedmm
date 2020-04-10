"""
.. module:: integrators
   :platform: Unix, Windows
   :synopsis: Unified Free Energy Dynamics Integrators

.. moduleauthor:: Charlles Abreu <abreu@eq.ufrj.br>

.. _Context: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.Context.html
.. _CustomCVForce: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.CustomCVForce.html
.. _CustomIntegrator: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.CustomIntegrator.html
.. _Force: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.Force.html
.. _System: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.System.html

"""

from simtk import openmm, unit


class CustomIntegrator(openmm.CustomIntegrator):
    """
    An extension of OpenMM's CustomIntegrator_ class with an extra per-dof variable `kT` whose
    content is the Boltzmann constant multiplied by the system temperature.

    Parameters
    ----------
        stepSize : float or unit.Quantity
            The step size with which to integrate the equations of motion.
        temperature : float or unit.Quantity
            The temperature.

    """

    def __init__(self, stepSize, temperature):
        super().__init__(stepSize)
        self.addPerDofVariable('kT', unit.MOLAR_GAS_CONSTANT_R*temperature)
