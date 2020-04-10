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
        stepSize : unit.Quantity
            The step size with which to integrate the equations of motion.
        drivingForce : :class:`~ufedmm.ufedmm.DrivingForce`
            The AFED driving force.

    """

    def __init__(self, stepSize, drivingForce):
        super().__init__(stepSize)
        self._driving_force = drivingForce
        self._per_parameter_variables = ['v', 'm', 'kT']
        for parameter in drivingForce._driver_parameters:
            self.addGlobalVariable(f'v_{parameter._name}', 0)
            self.addGlobalVariable(f'm_{parameter._name}', parameter._mass/parameter._mass_units)
            self.addGlobalVariable(f'kT_{parameter._name}', parameter._kT)
        self._has_conserved_energy = False

    def __repr__(self):
        # Human-readable version of each integrator step (adapted from choderalab/openmmtools)
        readable_lines = []

        readable_lines.append('Per-dof variables:')
        per_dof = []
        for index in range(self.getNumPerDofVariables()):
            per_dof.append(self.getPerDofVariableName(index))
        readable_lines.append('  ' + ', '.join(per_dof))

        readable_lines.append('Per-parameter variables:')
        per_parameter = set()
        for name in self._per_parameter_variables:
            values = []
            for parameter in self._driving_force._driver_parameters:
                per_parameter.add(f'{name}_{parameter._name}')
                values.append(self.getGlobalVariableByName(f'{name}_{parameter._name}'))
            readable_lines.append(f'  {name} = {values}')

        readable_lines.append('Global variables:')
        for index in range(self.getNumGlobalVariables()):
            name = self.getGlobalVariableName(index)
            if name not in per_parameter:
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
            'end',
        ]
        indent_level = 0
        for step in range(self.getNumComputations()):
            line = ''
            step_type, target, expr = self.getComputationStep(step)
            if step_type == 8:
                indent_level -= 1
            command = step_type_str[step_type].format(target=target, expr=expr)
            line += '{:4d}: '.format(step) + ' '*3*indent_level + command
            if step_type in [6, 7]:
                indent_level += 1
            readable_lines.append(line)
        return '\n'.join(readable_lines)

    def addPerParameterVariable(self, name, initialValue):
        """
        Defines a new per-driver-parameter variable.

        Parameters
        ----------
            variable : str
                The name of the per-driver-parameter variable.
            initialValue : unit.Quantity or list(unit.Quantity)
                The value initially assigned to the new variable, for all driver parameters. It can
                also be a list of values whose size matches the number of driver parameters.

        """

        try:
            for parameter, value in zip(self._driving_force._driver_parameters, initialValue):
                self.addGlobalVariable(f'{name}_{parameter._name}', value)
        except TypeError:
            for parameter in self._driving_force._driver_parameters:
                self.addGlobalVariable(f'{name}_{parameter._name}', initialValue)
        self._per_parameter_variables.append(name)

    def addComputePerParameter(self, variable, expression):
        """
        Add a step to the integration algorithm that computes a per-driver-parameter value.

        Parameters
        ----------
            variable : str
                The per-driver-parameter variable to store the computed value into.
            expression : str
                A mathematical expression involving both global and per-driver-parameter variables.
                In each integration step, its value is computed for every driver parameter and
                stored into the specified variable.

        Returns
        -------
            The index of the last step that was added.

        """

        def translate(expression, parameter):
            output = re.sub(r'\bx\b', f'{parameter}', expression)
            for symbol in self._per_parameter_variables:
                output = re.sub(r'\b{}\b'.format(symbol), f'{symbol}_{parameter}', output)
            output = re.sub(r'\bf([0-9]*)\b', f'(-deriv(energy\\1,{parameter}))', output)
            return output

        for parameter in self._driving_force._driver_parameters:
            name = parameter._name
            if variable == 'x':
                if parameter._period is not None:
                    # Apply periodic boundary conditions:
                    period = parameter._period/parameter._dimension
                    corrected_expression = f'select(step(-L/2-y),y+L,select(step(y-L/2),y-L,y))'
                    corrected_expression += f'; L={period}'
                    corrected_expression += f'; y={translate(expression, name)}'
                    self.addComputeGlobal(name, corrected_expression)
                else:
                    # Apply ellastic collision with hard wall:
                    self.addComputeGlobal(name, translate(expression, name))
                    for bound, op in zip([parameter._lower_bound, parameter._upper_bound], ['<', '>']):
                        if bound is not None:
                            limit = bound/parameter._dimension
                            self.beginIfBlock(f'{name} {op} {limit}')
                            self.addComputeGlobal(name, f'{2*limit}-{name}')
                            self.addComputeGlobal(f'v_{name}', f'-v_{name}')
                            self.endBlock()
            elif variable in self._per_parameter_variables:
                self.addComputeGlobal(f'{variable}_{name}', translate(expression, name))
            else:
                raise Exception('invalid per-parameter variable')

    def getPerParameterKineticEnergy(self):
        """
        Returns the kinetic energy of each driver parameter (in kJ/mole).

        """

        KE = []
        for parameter in self._driving_force._driver_parameters:
            m = self.getGlobalVariableByName(f'm_{parameter._name}')*parameter._mass_units
            v = self.getGlobalVariableByName(f'v_{parameter._name}')*parameter._dimension/unit.picoseconds
            KE.append(0.5*m*v*v)
        return np.array(KE)


class MassiveMiddleSchemeIntegrator(CustomIntegrator):
    """
    An abstract class aimed at facilitating the implementation of different AFED integrators that
    differ on the employed thermostat but share the following features:

    1. Integration of particle-related degrees of freedom is done by using a middle-type scheme
    :cite:`Zhang_2017` (i.e. kick-move-bath-move-kick), possibly involving multiple time stepping
    (RESPA) :cite:`Tuckerman_1992`.

    2. The system does not contain any holonomic constraints.

    3. Integration of driver parameters and their attached thermostats is done with a middle-type
    scheme as well, and is detached from the integration of all other dynamic variables, including
    the driver-parameter velocities.

    4. Integration of driver-parameter velocities is always done along with the integration of
    particle velocities, with possible splitting into multiple substeps by means of keyword argument
    `parameterLoops` (see below).

    Activation of multiple time scale integration (RESPA) is done by passing a list of integers
    through the keyword argument ``respaLoops`` (see below). The size of this list determines the
    number of considered time scales. Among the Force_ objects that belong to the simulated System_,
    only those whose force groups have been set to `k` will be considered at time scale `ḱ`. This
    includes the AFED-related :class:`~ufedmm.ufedmm.DrivingForce`.

    Parameters
    ----------
        temperature : unit.Quantity
            The temperature of the heat bath which the particles are attached to.
        stepSize : unit.Quantity
            The step size with which to integrate the system.
        drivingForce : :class:`~ufedmm.ufedmm.DrivingForce`
            The AFED driving force.

    Keyword Args
    ------------
        respaLoops : list(int), default=None
            A list of N integers, where ``respaLoops[k]`` determines how many iterations at time
            scale `k` are internally executed for every iteration at time scale `k+1`. If this is
            ``None``, then integration will take place at a single time scale.
        parameterLoops : int, default=1
            The number of loops with which to subdivide the integration of driver parameters and
            their attached thermostats.

    """

    def __init__(self, temperature, stepSize, drivingForce, respaLoops=None, parameterLoops=1):
        super().__init__(stepSize, drivingForce)
        kT = unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA*temperature
        self.addGlobalVariable('kT', kT)
        self._respaLoops = respaLoops
        if respaLoops is not None:
            for scale, nsteps in enumerate(respaLoops):
                if nsteps > 1:
                    self.addGlobalVariable(f'irespa{scale}', 0)
        self._parameterLoops = parameterLoops
        if parameterLoops > 1:
            self.addGlobalVariable('iparam', 0)
        self.addUpdateContextState()

    def _integrate_particles_respa(self, fraction, scale):
        n = self._respaLoops[scale]
        if n > 1:
            self.addComputeGlobal(f'irespa{scale}', '0')
            self.beginWhileBlock(f'irespa{scale} < {n}')
        if scale > 0:
            if scale == self._driving_force.getForceGroup():
                self._kick(fraction/(2*n), self.addComputePerParameter, scale)
            self._kick(fraction/(2*n), self.addComputePerDof, scale)
            self._integrate_particles_respa(fraction/n, scale-1)
            self._kick(fraction/(2*n), self.addComputePerDof, scale)
            if scale == self._driving_force.getForceGroup():
                self._kick(fraction/(2*n), self.addComputePerParameter, scale)
        else:
            self._inner_loop(fraction/n, group=0)
        if n > 1:
            self.addComputeGlobal(f'irespa{scale}', f'irespa{scale} + 1')
            self.endBlock()

    def _inner_loop(self, fraction, group):
        if group == '' or group == self._driving_force.getForceGroup():
            self._kick(fraction/2, self.addComputePerParameter, group)
        self._kick(fraction/2, self.addComputePerDof, group)
        self._move(fraction/2, self.addComputePerDof)
        self._bath(fraction, self.addComputePerDof)
        self._move(fraction/2, self.addComputePerDof)
        self._kick(fraction/2, self.addComputePerDof, group)
        if group == '' or group == self._driving_force.getForceGroup():
            self._kick(fraction/2, self.addComputePerParameter, group)

    def _move(self, fraction, addCompute):
        addCompute('x', f'x + {fraction}*dt*v')

    def _kick(self, fraction, addCompute, group=''):
        addCompute('v', f'v + {fraction}*dt*f{group}/m')

    def _bath(self, fraction, addCompute):
        pass

    def addIntegrateParticles(self, fraction):
        if self._respaLoops is None:
            self._inner_loop(fraction, group='')
        else:
            self._integrate_particles_respa(fraction, len(self._respaLoops)-1)

    def addIntegrateParameters(self, fraction):
        n = self._parameterLoops
        if n > 1:
            self.addComputeGlobal('iparam', '0')
            self.beginWhileBlock(f'iparam < {n}')
        self._move(0.5*fraction/n, self.addComputePerParameter)
        self._bath(fraction/n, self.addComputePerParameter)
        self._move(0.5*fraction/n, self.addComputePerParameter)
        if n > 1:
            self.addComputeGlobal('iparam', 'iparam + 1')
            self.endBlock()


class MassiveMiddleNHCIntegrator(MassiveMiddleSchemeIntegrator):
    """
    An AFED integrator based on a massive version of the Nosé-Hoover Chain thermostat
    :cite:`Martyna_1992`. This means that an independent thermostat chain is attached to each degree
    of freedom, including the AFED driver parameters. In this implementation, each chain is composed
    of two thermostats in series.

    All other properties of this integrator are inherited from :class:`MassiveMiddleSchemeIntegrator`.

    Parameters
    ----------
        temperature : unit.Quantity
            The temperature of the heat bath which the particles are attached to.
        timeScale : unit.Quantity
            The characteristic time scale of the thermostat chains.
        stepSize : unit.Quantity
            The step size with which to integrate the system.
        drivingForce : :class:`~ufedmm.ufedmm.DrivingForce`
            The AFED driving force.

    Keyword Args
    ------------
        respaLoops : list(int), default=None
            See :class:`MassiveMiddleSchemeIntegrator`.
        parameterLoops : int, default=1
            See :class:`MassiveMiddleSchemeIntegrator`.
        conservedEnergy : bool, default=False
            Whether to integrate thermostat coordinates so that one can compute the
            thermostat-related part of the non-Hamiltonian conserved energy.

    """

    def __init__(self, temperature, timeScale, stepSize, drivingForce, **kwargs):
        conserved_energy = kwargs.pop('conservedEnergy', False)
        super().__init__(temperature, stepSize, drivingForce, **kwargs)
        self._has_conserved_energy = conserved_energy
        kT = unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA*temperature
        Q = kT*timeScale**2
        Qparams = [param._kT*timeScale**2 for param in self._driving_force._driver_parameters]
        for i in range(2):
            self.addGlobalVariable(f'Q{i+1}', Q)
            self.addPerDofVariable(f'v{i+1}', 0)
            self.addPerParameterVariable(f'Q{i+1}', Qparams)
            self.addPerParameterVariable(f'v{i+1}', 0)
            if self._has_conserved_energy:
                self.addPerDofVariable(f'eta{i+1}', 0)
                self.addPerParameterVariable(f'eta{i+1}', 0)

        self.addIntegrateParticles(0.5)
        self.addIntegrateParameters(1)
        self.addIntegrateParticles(0.5)

    def _bath(self, fraction, addCompute):
        addCompute('v2', f'v2 + {fraction/2}*dt*(Q1*v1^2 - kT)/Q2')
        addCompute('v1', f'v1*exp(-Dt*v2) + Dt*(m*v^2 - kT)/Q1; Dt={fraction/2}*dt')
        addCompute('v', f'v*exp(-{fraction}*dt*v1)')
        if self._has_conserved_energy:
            addCompute('eta1', f'eta1 + {fraction}*dt*v1')
            addCompute('eta2', f'eta2 + {fraction}*dt*v2')
        addCompute('v1', f'(v1 + Dt*(m*v^2 - kT)/Q1)*exp(-Dt*v2); Dt={fraction/2}*dt')
        addCompute('v2', f'v2 + {fraction/2}*dt*(Q1*v1^2 - kT)/Q2')

    def getThermostatEnergy(self):
        """
        Returns the total energy of all thermostats (in kJ/mole). If ``thermoCoordinates=False``,
        then only the kinetic part is computed.

        """
        energy = 0.0
        for i in range(2):
            Q = self.getGlobalVariableByName(f'Q{i+1}')
            velocities = self.getPerDofVariableByName(f'v{i+1}')
            energy += 0.5*Q*sum(v[0]**2 + v[1]**2 + v[2]**2 for v in velocities)
            if self._has_conserved_energy:
                kT = self.getGlobalVariableByName('kT')
                coordinates = self.getPerDofVariableByName(f'eta{i+1}')
                energy += kT*sum(eta[0] + eta[1] + eta[2] for eta in coordinates)
            for parameter in self._driving_force._driver_parameters:
                Q = self.getGlobalVariableByName(f'Q{i+1}_{parameter._name}')
                v = self.getGlobalVariableByName(f'v{i+1}_{parameter._name}')
                energy += 0.5*Q*v**2
                if self._has_conserved_energy:
                    kT = self.getGlobalVariableByName(f'kT_{parameter._name}')
                    eta = self.getGlobalVariableByName(f'eta{i+1}_{parameter._name}')
                    energy += kT*eta
        return energy*unit.kilojoules_per_mole
