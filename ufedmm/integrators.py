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
    An extension of OpenMM's CustomIntegrator_ class with an extra per-dof variable named `kT`,
    whose content is the Boltzmann constant multiplied by the system temperature. A per-dof
    temperature is necessary if the extended-space variables and the physical system are coupled
    adiabatically to thermostats at different temperatures. Otherwise, any other OpenMM integrator
    can be used.

    Parameters
    ----------
        temperature : float or unit.Quantity
            The temperature.
        step_size : float or unit.Quantity
            The step size with which to integrate the equations of motion.

    """

    def __init__(self, temperature, step_size):
        super().__init__(step_size)
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


class DoubleTimeScaleRegulatedIntegrator(CustomIntegrator):
    """
    A regulated version of the massive Nose-Hoover-Langevin :cite:`Samoletov_2007,Leimkuhler_2009`
    method. Regulation means that velocities are modified so as to remain below a
    temperature-dependent speed limit. This method is closely related to the SIN(R) method
    :cite:`Leimkuhler_2013` and allows multiple time-scale integration without resonance.

    The following :term:`SDE` system is solved for every degree of freedom in the system:

    .. math::
        & dr_i = v_i dt \\\\
        & dp_i = F_i dt - v_{\\eta_i} m_i v_i dt \\\\
        & dv_{\\eta_i} = \\frac{1}{Q}\\left(\\frac{n+1}{n} m_i v_i^2 - k_B T\\right) dt
                - \\gamma v_{\\eta_i} dt + \\sqrt{\\frac{2\\gamma k_B T}{Q}} dW_i,

    where:

    .. math::
        v_i = c_i \\tanh\\left(\\frac{p_i}{m_i c_i}\\right).

    Here, :math:`c_i = \\sqrt{\\frac{n k T}{m_i}}` is speed limit for such degree of freedom.
    As usual, the inertial parameter :math:`Q` is defined as :math:`Q = k_B T \\tau^2`, with
    :math:`\\tau` being a relaxation time :cite:`Tuckerman_1992`. An approximate solution is
    obtained by applying the Trotter-Suzuki splitting formula:

    .. math::
        e^{\\Delta t\\mathcal{L}} =
        e^{\\frac{\\Delta t}{2}\\mathcal{L}^1_p}
        \\left[e^{\\frac{\\delta t}{2}\\mathcal{L}^0_p}
        e^{\\frac{\\delta t}{2}\\mathcal{L}_r}
        e^{\\delta t \\mathcal{L}_\\mathrm{bath}}
        e^{\\frac{\\delta t}{2}\\mathcal{L}_r}
        e^{\\frac{\\delta t}{2}\\mathcal{L}^0_p}\\right]^m
        e^{\\frac{\\Delta t}{2}\\mathcal{L}^1_p}

    where :math:`\\delta t = \\frac{\\Delta t}{m}`. Each exponential operator above is the solution
    of a differential equation.

    The exact solution for the physical-system part is:

    .. math::
        r_i(t) = r_i^0 + c_i \\mathrm{tanh}\\left(\\frac{p_i}{m c_i}\\right) t

    .. math::
        p_i(t) = p_i^0 + F_i t

    The bath propagator is further split as:

    .. math::
        e^{\\delta t \\mathcal{L}_\\mathrm{bath}} =
        e^{\\frac{\\delta t}{2m}\\mathcal{L}_B}
        e^{\\frac{\\delta t}{2m}\\mathcal{L}_S}
        e^{\\frac{\\delta t}{m}\\mathcal{L}_O}
        e^{\\frac{\\delta t}{2m}\\mathcal{L}_S}
        e^{\\frac{\\delta t}{2m}\\mathcal{L}_B}

    Part 'B' is a boost, whose solution is:

    .. math::
        v_{\\eta_i}(t) = v_{\\eta_i}^0 +
                         \\frac{1}{Q}\\left(\\frac{n+1}{n} m_i v_i^2 - k_B T\\right) t

    Part 'S' is a scaling, whose solution is:

    .. math::
        p_i(t) = m_i c_i \\mathrm{arcsinh}\\left[
                    \\sinh\\left(\\frac{p_i}{m_i c_i}\\right) e^{- v_{\\eta_i} t}
                 \\right]

    Part 'O' is an Ornstein–Uhlenbeck process, whose solution is:

    .. math::
        v_{\\eta_i}(t) = v_{\\eta_i}^0 e^{-\\gamma t}
                   + \\sqrt{\\frac{k_B T}{Q}(1-e^{-2\\gamma t})} R_N

    where :math:`R_N` is a normally distributed random number.

    Parameters
    ----------
        step_size : float or unit.Quantity
            The outer step size with which to integrate the equations of motion.
        loops : int
            The number of internal substeps at each time step.
        temperature : unit.Quantity
            The temperature of the heat bath.
        time_scale : unit.Quantity (time)
            The relaxation time (:math:`\\tau`) of the Nose-Hoover thermostat.
        friction_coefficient : unit.Quantity (1/time)
            The friction coefficient (:math:`\\gamma`) of the Langevin thermostat.
        n : int or float
            The regulation parameter.

    """

    def __init__(self, step_size, substeps, temperature, time_scale, friction_coefficient, n):
        super().__init__(temperature, step_size)
        self.addGlobalVariable('omega', 1/time_scale)
        self.addGlobalVariable('friction', friction_coefficient)
        self.addPerDofVariable('v_eta', 0)

        fraction = 1.0/substeps

        boost = f'v_eta + G*{0.5*fraction}*dt'
        boost += f'; G=({(n+1)/n}*m*(c*tanh(v/c))^2 - kT)/Q'
        boost += f'; c=sqrt({n}*kT/m)'
        boost += '; Q=kT/omega^2'

        scaling = 'c*asinhz'
        scaling += '; asinhz=(2*step(z)-1)*log(select(step(za-1E8),2*za,za+sqrt(1+z*z))); za=abs(z)'
        scaling += f'; z=sinh(v/c)*exp(-v_eta*{0.5*fraction}*dt)'
        scaling += f'; c=sqrt({n}*kT/m)'

        Ornstein_Uhlenbeck = 'v_eta*z + omega*sqrt(1-z^2)*gaussian'
        Ornstein_Uhlenbeck += f'; z=exp(-friction*{fraction}*dt)'

        self.setKineticEnergyExpression(f'0.5*m*(c*tanh(v/c))^2; c=sqrt({n}*kT/m)')
        self.addComputePerDof('v', 'v + 0.5*dt*f1/m')
        for i in range(substeps):
            self.addComputePerDof('v', f'v + {0.5*fraction}*dt*f0/m')
            self.addComputePerDof('x', f'x + c*tanh(v/c)*{0.5*fraction}*dt; c=sqrt({n}*kT/m)')
            self.addComputePerDof('v_eta', boost)
            self.addComputePerDof('v', scaling)
            self.addComputePerDof('v_eta', Ornstein_Uhlenbeck)
            self.addComputePerDof('v', scaling)
            self.addComputePerDof('v_eta', boost)
            self.addComputePerDof('x', f'x + c*tanh(v/c)*{0.5*fraction}*dt; c=sqrt({n}*kT/m)')
            self.addComputePerDof('v', f'v + {0.5*fraction}*dt*f0/m')
        self.addComputePerDof('v', 'v + 0.5*dt*f1/m')


class GeodesicLangevinIntegrator(CustomIntegrator):
    """
    A geodesic Langevin integrator :cite:`Leimkuhler_2016`, which can be integrated by using
    either the LF-Middle or the VV-Middle scheme :cite:`Zhang_2019`.

    .. note:
        The VV-Middle scheme is also known as the BAOAB :cite:`Leimkuhler_2016` or VRORV method.

    Parameters
    ----------
        temperature : float or unit.Quantity
            The temperature.
        friction_coefficient : float or unit.Quantity
            The friction coefficient.
        step_size : float or unit.Quantity
            The time-step size.

    Keyword Args
    ------------
        scheme : str, default='LF-Middle'
            The integration scheme. Valid options are 'LF-Middle' (default) and 'VV-Middle'.
        rattles : int, default=1
            The number of RATTLE computations. If `rattles=0`, then no constraints are considered.

    Example
    -------
        >>> import ufedmm
        >>> dt = 2*unit.femtoseconds
        >>> temp = 300*unit.kelvin
        >>> gamma = 10/unit.picoseconds
        >>> ufedmm.GeodesicLangevinIntegrator(temp, gamma, dt, rattles=1, scheme='VV-Middle')
        Per-dof variables:
          kT, x0
        Global variables:
          friction = 10.0
        Computation steps:
           0: allow forces to update the context state
           1: v <- v + 0.5*dt*f/m
           2: constrain velocities
           3: x <- x + 0.5*dt*v
           4: x0 <- x
           5: constrain positions
           6: v <- v + (x - x0)/(0.5*dt)
           7: constrain velocities
           8: v <- z*v + sqrt((1 - z*z)*kT/m)*gaussian; z = exp(-friction*dt)
           9: constrain velocities
          10: x <- x + 0.5*dt*v
          11: x0 <- x
          12: constrain positions
          13: v <- v + (x - x0)/(0.5*dt)
          14: constrain velocities
          15: v <- v + 0.5*dt*f/m
          16: constrain velocities

    """

    def __init__(self, temperature, friction_coefficient, step_size, rattles=1, scheme='LF-Middle'):
        if scheme not in ['LF-Middle', 'VV-Middle']:
            raise Exception(f'Invalid value {scheme} for keyword scheme')
        super().__init__(temperature, step_size)
        self._rattles = rattles
        self.addGlobalVariable('friction', friction_coefficient)
        if rattles > 1:
            self.addGlobalVariable('irattle', 0)
        self.addPerDofVariable('x0', 0)
        self.addUpdateContextState()
        self._B(0.5 if scheme == 'VV-Middle' else 1)
        self._A()
        self._O()
        self._A()
        self._B(0.5 if scheme == 'VV-Middle' else 0)

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

    def _B(self, fraction):
        if fraction > 0:
            self.addComputePerDof('v', f'v + {fraction}*dt*f/m')
            if self._rattles > 0:
                self.addConstrainVelocities()

    def _O(self):
        expression = 'z*v + sqrt((1 - z*z)*kT/m)*gaussian; z = exp(-friction*dt)'
        self.addComputePerDof('v', expression)
        if self._rattles > 0:
            self.addConstrainVelocities()
