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

import re

import numpy as np

from simtk import openmm, unit


class CustomIntegrator(openmm.CustomIntegrator):
    """
    An extension of OpenMM's CustomIntegrator_ class. This extension facilitates the specification
    of variables and computation steps in a per-driver-parameter fashion. These computations are
    defined in the same manner as per-dof computations in the original class.

    .. note::
        For every driver parameter in `drivingForce`, per-driver-parameters `v` (velocity), `m`
        (mass), and `kT` (Boltzmann constant times temperature) are automatically created, as well
        as read-only force variables `f`, `f0`, `f1`, and so on (see CustomIntegrator_).

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
