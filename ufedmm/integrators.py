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

    def __repr__(self):
        """
        A human-readable version of each integrator step (adapted from openmmtools)

        Returns
        -------
        readable_lines : str
           A list of human-readable versions of each step of the integrator

        """
        readable_lines = []

        readable_lines.append('Per-dof variables:')
        per_dof = []
        for index in range(self.getNumPerDofVariables()):
            per_dof.append(self.getPerDofVariableName(index))
        readable_lines.append('  ' + ', '.join(per_dof))

        readable_lines.append('Global variables:')
        for index in range(self.getNumGlobalVariables()):
            name = self.getGlobalVariableName(index)
            value = self.getGlobalVariable(index)
            readable_lines.append(f'  {name} = {value}')

        readable_lines.append('Computation steps:')

        step_type_str = [
            '{target} <- {expr}',
            '{target} <- {expr}',
            '{target} <- sum({expr})',
            'constrain positions',
            'constrain velocities',
            'allow forces to update the context state',
            'if ({expr}):',
            'while ({expr}):',
            'end'
        ]
        indent_level = 0
        for step in range(self.getNumComputations()):
            line = ''
            step_type, target, expr = self.getComputationStep(step)
            if step_type == 8:
                indent_level -= 1
            command = step_type_str[step_type].format(target=target, expr=expr)
            line += '{:4d}: '.format(step) + '   '*indent_level + command
            if step_type in [6, 7]:
                indent_level += 1
            readable_lines.append(line)
        return '\n'.join(readable_lines)


class GeodesicBAOABIntegrator(CustomIntegrator):
    """
    Geodesic BAOAB integrator.

    Parameters
    ----------
        timestep : float or unit.Quantity
            The time-step size.
        temperature : float or unit.Quantity
            The temperature.
        friction_coefficient : float or unit.Quantity
            The friction coefficient.
    
    Keyword Args
    ------------
        rattles : int, default=1
            The number of RATTLE computations. If `rattles=0`, then no constraints are considered.

    """
    def __init__(self, timestep, temperature, friction_coefficient, rattles=1):
        super().__init__(timestep, temperature)
        self._rattles = rattles
        self.addGlobalVariable('friction', friction_coefficient)
        if rattles > 1:
            self.addGlobalVariable('irattle', 0)
        self.addPerDofVariable('x0', 0)
        self.addUpdateContextState()
        self._B()
        self._A()
        self._O()
        self._A()
        self._B()

    def _A(self):
        if self._rattles > 1:
            self.addComputeGlobal('irattle', '0')
            self.beginWhileBlock(f'irattle < {self._rattles}')
        self.addComputePerDof('x', f'x + {0.5/max(1, self._rattles)}*dt*v')
        if self._rattles > 0:
            self.addComputePerDof('x0', 'x')
            self.addConstrainPositions()
            self.addComputePerDof('v', f'v + (x - x0)/({0.5/self._rattles}*dt)')
            self.addConstrainVelocities()
        if self._rattles > 1:
            self.addComputeGlobal('irattle', 'irattle + 1')
            self.endBlock()

    def _B(self):
        self.addComputePerDof('v', 'v + 0.5*dt*f/m')
        if self._rattles > 0:
            self.addConstrainVelocities()

    def _O(self):
        expression = 'z*v + sqrt((1 - z*z)*kT/m)*gaussian; z = exp(-friction*dt)'
        self.addComputePerDof('v', expression)
        if self._rattles > 0:
            self.addConstrainVelocities()
