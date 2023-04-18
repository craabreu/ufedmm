"""
.. module:: integrators
   :platform: Unix, Windows
   :synopsis: Unified Free Energy Dynamics Integrators

.. moduleauthor:: Charlles Abreu <abreu@eq.ufrj.br>

.. _Context:
    http://docs.openmm.org/latest/api-python/generated/openmm.openmm.Context.html
.. _CustomCVForce:
    http://docs.openmm.org/latest/api-python/generated/openmm.openmm.CustomCVForce.html
.. _CustomIntegrator:
    http://docs.openmm.org/latest/api-python/generated/openmm.openmm.CustomIntegrator.html
.. _Force:
    http://docs.openmm.org/latest/api-python/generated/openmm.openmm.Force.html
.. _NonbondedForce:
    http://docs.openmm.org/latest/api-python/generated/openmm.openmm.NonbondedForce.html
.. _System:
    http://docs.openmm.org/latest/api-python/generated/openmm.openmm.System.html

"""

import numpy as np
import openmm
from openmm import unit

from ufedmm.ufedmm import _standardized


def add_inner_nonbonded_force(system, inner_switch, inner_cutoff, force_group_index):
    """
    To a given OpenMM System_ containing a NonbondedForce_ object, this function adds a new force
    group with the purpose of performing multiple time-scale integration according to the RESPA2
    splitting scheme of Morrone, Zhou, and Berne :cite:`Morrone_2010`. Besides, it assigns the
    provided `force_group_index` to this new group and `force_group_index+1` to the original
    NonbondedForce_. When used in any instance of :class:`AbstractMiddleRespaIntegrator`, the new
    force group must be identified as being embodied by the NonbondedForce_ as opposed to being
    complimentary to it.

    .. warning:
        The new force group is not intended to contribute to the system energy. Its sole purpose
        is to provide a smooth, short-range force calculator for some intermediary time scale in
        a RESPA-type integration.

    Parameters
    ----------
        system : openmm.System
            The system the inner force will be added to, which must contain a NonbondedForce_.
        inner_switch : float or unit.Quantity
            The inner switching distance, where the interaction of an atom pair begins to switch
            off to zero.
        inner_cutoff : float or unit.Quantity
            The inner cutoff distance, where the interaction of an atom pairs completely switches
            off.
        force_group_index : int
            The force group the new interactions will belong to. The old NonbondedForce_ will be
            automatically assigned to `force_group_index+1`.

    Example
    -------
        >>> import ufedmm
        >>> from openmm import unit
        >>> dt = 2*unit.femtoseconds
        >>> temp = 300*unit.kelvin
        >>> tau = 10*unit.femtoseconds
        >>> gamma = 10/unit.picoseconds
        >>> model = ufedmm.AlanineDipeptideModel()
        >>> ufedmm.add_inner_nonbonded_force(model.system, 5*unit.angstroms, 8*unit.angstroms, 1)

    """
    if openmm.__version__ < "7.5":
        raise Exception("add_inner_nonbonded_force requires OpenMM version >= 7.5")
    try:
        nonbonded_force = next(
            filter(lambda f: isinstance(f, openmm.NonbondedForce), system.getForces())
        )
    except StopIteration:
        raise Exception("add_inner_nonbonded_force requires system with NonbondedForce")
    if (
        nonbonded_force.getNumParticleParameterOffsets() > 0
        or nonbonded_force.getNumExceptionParameterOffsets() > 0
    ):
        raise Exception("add_inner_nonbonded_force does not support parameter offsets")
    periodic = nonbonded_force.usesPeriodicBoundaryConditions()
    rs = _standardized(inner_switch)
    rc = _standardized(inner_cutoff)
    a = rc + rs
    b = rc * rs
    c = (30 / (rc - rs) ** 5) * np.array([b**2, -2 * a * b, a**2 + 2 * b, -2 * a, 1])
    f0s = sum([c[n] * rs ** (n + 1) / (n + 1) for n in range(5)])

    def coeff(n, m):
        return c[m - 1] if m == n else c[m - 1] / (m - n)

    def func(n, m):
        return "*log(r)" if m == n else (f"*r^{m-n}" if m > n else f"/r^{n-m}")

    def val(n, m):
        return f0s if m == 0 else (coeff(n, m) - coeff(0, m) if n != m else coeff(n, m))

    def sgn(n, m):
        return "+" if m > 0 and val(n, m) >= 0 else ""

    def S(n):
        return "".join(f"{sgn(n, m)}{val(n, m)}{func(n, m)}" for m in range(6))

    potential = "eps4*((sigma/r)^12-(sigma/r)^6)+Qprod/r"
    potential += f"+step(r-{rs})*(eps4*(sigma^12*({S(12)})-sigma^6*({S(6)}))+Qprod*({S(1)}))"
    mixing_rules = "; Qprod=Q1*Q2"
    mixing_rules += "; sigma=halfsig1+halfsig2"
    mixing_rules += "; eps4=sqrt4eps1*sqrt4eps2"
    force = openmm.CustomNonbondedForce(potential + mixing_rules)
    for parameter in ["Q", "halfsig", "sqrt4eps"]:
        force.addPerParticleParameter(parameter)
    force.setNonbondedMethod(force.CutoffPeriodic if periodic else force.CutoffNonPeriodic)
    force.setCutoffDistance(inner_cutoff)
    force.setUseLongRangeCorrection(False)
    ONE_4PI_EPS0 = 138.93545764438198
    for index in range(nonbonded_force.getNumParticles()):
        charge, sigma, epsilon = map(_standardized, nonbonded_force.getParticleParameters(index))
        force.addParticle([charge * np.sqrt(ONE_4PI_EPS0), sigma / 2, np.sqrt(4 * epsilon)])
    non_exclusion_exceptions = []
    for index in range(nonbonded_force.getNumExceptions()):
        i, j, q1q2, sigma, epsilon = nonbonded_force.getExceptionParameters(index)
        q1q2, sigma, epsilon = map(_standardized, [q1q2, sigma, epsilon])
        force.addExclusion(i, j)
        if q1q2 != 0.0 or epsilon != 0.0:
            non_exclusion_exceptions.append((i, j, q1q2 * ONE_4PI_EPS0, sigma, 4 * epsilon))
    force.setForceGroup(force_group_index)
    system.addForce(force)
    if non_exclusion_exceptions:
        exceptions = openmm.CustomBondForce(f"step({rc}-r)*({potential})")
        for parameter in ["Qprod", "sigma", "eps4"]:
            exceptions.addPerBondParameter(parameter)
        for i, j, Qprod, sigma, eps4 in non_exclusion_exceptions:
            exceptions.addBond(i, j, [Qprod, sigma, eps4])
        exceptions.setForceGroup(force_group_index)
        system.addForce(exceptions)
    nonbonded_force.setForceGroup(force_group_index + 1)


class CustomIntegrator(openmm.CustomIntegrator):
    """
    An extension of OpenMM's CustomIntegrator_ class with an extra per-dof variable named
    `temperature`, whose content is the temperature of the heat bath associated to each
    degree of freedom. A per-dof temperature is necessary if the extended-space variables
    and the physical system are coupled adiabatically to thermostats at different temperatures.
    Otherwise, any other OpenMM integrator can be used.

    Parameters
    ----------
        temperature : float or unit.Quantity
            The temperature.
        step_size : float or unit.Quantity
            The step size with which to integrate the equations of motion.

    """

    def __init__(self, temperature, step_size):
        super().__init__(step_size)
        self.temperature = temperature
        self.addPerDofVariable("kT", unit.MOLAR_GAS_CONSTANT_R * temperature)
        self._up_to_date = False

    def __repr__(self):
        """
        A human-readable version of each integrator step (adapted from openmmtools)

        Returns
        -------
        readable_lines : str
           A list of human-readable versions of each step of the integrator

        """
        readable_lines = []

        self.getNumPerDofVariables() > 0 and readable_lines.append("Per-dof variables:")
        per_dof = []
        for index in range(self.getNumPerDofVariables()):
            per_dof.append(self.getPerDofVariableName(index))
        readable_lines.append("  " + ", ".join(per_dof))

        self.getNumGlobalVariables() > 0 and readable_lines.append("Global variables:")
        for index in range(self.getNumGlobalVariables()):
            name = self.getGlobalVariableName(index)
            value = self.getGlobalVariable(index)
            readable_lines.append(f"  {name} = {value}")

        readable_lines.append("Computation steps:")

        step_type_str = [
            "{target} <- {expr}",
            "{target} <- {expr}",
            "{target} <- sum({expr})",
            "constrain positions",
            "constrain velocities",
            "allow forces to update the context state",
            "if ({expr}):",
            "while ({expr}):",
            "end",
        ]
        indent_level = 0
        for step in range(self.getNumComputations()):
            line = ""
            step_type, target, expr = self.getComputationStep(step)
            if step_type == 8:
                indent_level -= 1
            command = step_type_str[step_type].format(target=target, expr=expr)
            line += "{:4d}: ".format(step) + "   " * indent_level + command
            if step_type in [6, 7]:
                indent_level += 1
            readable_lines.append(line)
        return "\n".join(readable_lines)

    def update_temperatures(self, system_temperature, extended_space_temperatures):
        nparticles = len(self.getPerDofVariableByName("kT")) - len(extended_space_temperatures)
        temperatures = [system_temperature] * nparticles + extended_space_temperatures
        kT = [unit.MOLAR_GAS_CONSTANT_R * T * openmm.Vec3(1, 1, 1) for T in temperatures]
        self.setPerDofVariableByName("kT", kT)
        self._up_to_date = True

    def step(self, steps):
        if not self._up_to_date:
            self.update_temperatures(self.temperature, [])
        super().step(steps)


class AbstractMiddleRespaIntegrator(CustomIntegrator):
    """
    An abstract class for middle-type, multiple time-scale integrators.

    .. warning::
        This class is meant for inheritance only and does not actually include thermostatting.
        Concrete subclasses are available, such as :class:`MiddleMassiveNHCIntegrator` and
        :class:`GeodesicLangevinIntegrator`, for instance.

    Child classes will differ by the thermostat algorithm, which must be implemented
    by overriding the `_bath` method (see the example below).
    Temperature is treated as a per-dof parameter so as to allow adiabatic simulations.

    The following :term:`ODE` system is solved for every degree of freedom in the system,
    with possibly :math:`n_c` holonomic constraints and with forces possibly split into
    :math:`m` parts according to their characteristic time scales:

    .. math::
        & \\dot{r}_i = v_i \\\\
        & \\dot{v}_i = \\frac{\\sum_{k=1}^m F_i^{[k]}}{m_i}
                     + \\sum_{k=1}^{n_c} \\lambda_k \\nabla_{r_i} \\sigma_k
                     + \\mathrm{bath}(T_i, v_i) \\\\
        & \\sigma_k(\\mathbf{r}) = 0

    An approximate solution is obtained by applying the Trotter-Suzuki splitting formula.
    In the particular case of two time scales, the default splitting scheme goes as follows:

    .. math::
        e^{\\Delta t\\mathcal{L}} =
            e^{\\frac{\\Delta t}{2}\\mathcal{L}^{[1]}_v}
            \\left[
                e^{\\frac{\\Delta t}{2 n_0}\\mathcal{L}^{[0]}_v}
                \\left(
                    e^{\\frac{\\Delta t}{2 n_0 n_b}\\mathcal{L}_r}
                    e^{\\frac{\\Delta t}{n_0 n_b}\\mathcal{L}_\\mathrm{bath}}
                    e^{\\frac{\\Delta t}{2 n_0 n_b}\\mathcal{L}_r}
                \\right)^{n_b}
                e^{\\frac{\\Delta t}{2 n_0}\\mathcal{L}^{[0]}_v}
            \\right]^{n_0}
            e^{\\frac{\\Delta t}{2}\\mathcal{L}^{[1]}_v}

    Each exponential operator is the solution of a particular subsystem of equations.
    If :math:`\\mathrm{bath}(T_i, v_i) = 0`, the scheme above is time-reversible,
    measure-preserving, and symplectic. It is referred to as the ``VV-Middle`` scheme
    :cite:`Zhang_2019`, where VV stands for Velocity Verlet. An alternative approach
    is also available, which is:

    .. math::
        e^{\\Delta t\\mathcal{L}} =
            \\left[
                \\left(
                    e^{\\frac{\\Delta t}{2 n_0 n_b}\\mathcal{L}_r}
                    e^{\\frac{\\Delta t}{n_0 n_b}\\mathcal{L}_\\mathrm{bath}}
                    e^{\\frac{\\Delta t}{2 n_0 n_b}\\mathcal{L}_r}
                \\right)^{n_b}
                e^{\\frac{\\Delta t}{n_0}\\mathcal{L}^{[0]}_v}
            \\right]^{n_0}
            e^{\\Delta t \\mathcal{L}^{[1]}_v}

    This is referred to as the ``LF-Middle`` scheme :cite:`Zhang_2019`, where LF stands for
    Leap-Frog. In contrast to the previous scheme, it is not time-reversible. However, in single
    time-scale simulations, the two approaches result in equivalent coordinate trajectories,
    while the latter provides a velocity trajectory more consistent with the Maxwell-Boltzmann
    distribution at the specified temperature :cite:`Zhang_2019`.

    Parameters
    ----------
        temperature : unit.Quantity
            The temperature of the heat bath.
        step_size : float or unit.Quantity
            The outer step size with which to integrate the equations of motion.

    Keyword Args
    ------------
        num_rattles : int, default=0
            The number of RATTLE computations for geodesic integration :cite:`Leimkuhler_2016`.
            If ``num_rattles=0``, then no constraints are considered at all.
        scheme : str, default='VV-Middle'
            Which splitting scheme will be used. Valid options are 'VV-Middle' and 'LF-Middle'.
        respa_loops : list(int), default=[1]
            A list of `m` integers, where `respa_loops[k]` determines how many substeps with
            force group `k` are internally executed for every step with force group `k+1`.
        bath_loops : int, default=1
            The number of iterations of the bath operator per each step at time scale `0`. This
            is useful when the bath operator is not exact, but derived from a splitting solution.
        embodied_force_groups : list(int), default=[]
            A list of indices of force groups. The presence of an index `i` is this list means that
            the contribution of force group `i` is embodied in force group `i+1`. Therefore, such
            contribution must be properly subtracted during the integration at time scale `i+1`.
            This feature requires OpenMM 7.5 or a newer version.
        unroll_loops : bool, default=True
            Whether the integrator loops should be unrolled for improving efficiency. Using
            ``unroll_loops=False`` can be useful for printing the integrator steps.

    Example
    -------
        >>> from ufedmm import integrators
        >>> from openmm import unit
        >>> class MiddleNoseHooverIntegrator(integrators.AbstractMiddleRespaIntegrator):
        ...     def __init__(self, ndof, tau, temperature, step_size, num_rattles=1):
        ...         super().__init__(temperature, step_size, num_rattles, 'VV-Middle', [1], 1)
        ...         kB = 8.3144626E-3*unit.kilojoules_per_mole/unit.kelvin
        ...         gkT = ndof*unit.MOLAR_GAS_CONSTANT_R*temperature
        ...         self.addGlobalVariable('gkT', gkT)
        ...         self.addGlobalVariable('Q', gkT*tau**2)
        ...         self.addGlobalVariable('v_eta', 0)
        ...         self.addGlobalVariable('twoK', 0)
        ...         self.addGlobalVariable('scaling', 1)
        ...     def _bath(self, fraction):
        ...         self.addComputeSum('twoK', 'm*v*v')
        ...         self.addComputeGlobal(
        ...             'v_eta', f'v_eta + {0.5*fraction}*dt*(twoK - gkT)/Q'
        ...         )
        ...         self.addComputeGlobal('scaling', f'exp(-{fraction}*dt*v_eta)')
        ...         self.addComputePerDof('v', f'v*scaling')
        ...         self.addComputeGlobal(
        ...             'v_eta', f'v_eta + {0.5*fraction}*dt*(scaling^2*twoK - gkT)/Q'
        ...         )
        >>> integrator = MiddleNoseHooverIntegrator(
        ...     500, 10*unit.femtoseconds, 300*unit.kelvin,
        ...     1*unit.femtoseconds, num_rattles=0
        ... )
        >>> print(integrator)
        Per-dof variables:
          kT
        Global variables:
          gkT = 1247.169392722986
          Q = 0.1247169392722986
          v_eta = 0.0
          twoK = 0.0
          scaling = 1.0
        Computation steps:
           0: allow forces to update the context state
           1: v <- v + 0.5*dt*f/m
           2: x <- x + 0.5*dt*v
           3: twoK <- sum(m*v*v)
           4: v_eta <- v_eta + 0.5*dt*(twoK - gkT)/Q
           5: scaling <- exp(-1.0*dt*v_eta)
           6: v <- v*scaling
           7: v_eta <- v_eta + 0.5*dt*(scaling^2*twoK - gkT)/Q
           8: x <- x + 0.5*dt*v
           9: v <- v + 0.5*dt*f/m

    """

    def __init__(
        self,
        temperature,
        step_size,
        num_rattles=0,
        scheme="VV-Middle",
        respa_loops=[1],
        bath_loops=1,
        intertwine=True,
        embodied_force_groups=[],
        unroll_loops=True,
    ):
        if scheme not in ["LF-Middle", "VV-Middle"]:
            raise Exception(f"Invalid value {scheme} for keyword scheme")
        super().__init__(temperature, step_size)
        self._num_rattles = num_rattles
        self._scheme = scheme
        self._respa_loops = respa_loops
        self._bath_loops = bath_loops
        self._intertwine = intertwine
        self._subtractive_groups = embodied_force_groups
        num_rattles > 0 and self.addPerDofVariable("x0", 0)
        num_rattles > 1 and self.addGlobalVariable("irattle", 0)
        if not unroll_loops:
            for scale, n in enumerate(respa_loops):
                n > 1 and self.addGlobalVariable(f"irespa{scale}", 0)
            bath_loops > 1 and self.addGlobalVariable("ibath", 0)
        if embodied_force_groups:
            if openmm.__version__ < "7.5":
                raise Exception("Use of `embodied_force_groups` option requires OpenMM >= 7.5")
            self.addPerDofVariable("f_emb", 0)
            integration_groups = set(range(len(respa_loops))) - set(embodied_force_groups)
            self.setIntegrationForceGroups(integration_groups)

        self.addUpdateContextState()
        self._step_initialization()
        if unroll_loops:
            self._integrate_respa_unrolled(1, len(respa_loops) - 1)
        else:
            self._integrate_respa(1, len(respa_loops) - 1)

    def _step_initialization(self):
        pass

    def _integrate_respa(self, fraction, scale):
        if scale >= 0:
            n = self._respa_loops[scale]
            if n > 1:
                self.addComputeGlobal(f"irespa{scale}", "0")
                self.beginWhileBlock(f"irespa{scale} < {n-1/2}")
            self._boost(fraction / (2 * n if self._scheme == "VV-Middle" else n), scale)
            self._integrate_respa(fraction / n, scale - 1)
            self._scheme == "VV-Middle" and self._boost(fraction / (2 * n), scale)
            if n > 1:
                self.addComputeGlobal(f"irespa{scale}", f"irespa{scale} + 1")
                self.endBlock()
        else:
            self._intertwine or self._translation(0.5 * fraction)
            n = self._bath_loops
            if n > 1:
                self.addComputeGlobal("ibath", "0")
                self.beginWhileBlock(f"ibath < {n-1/2}")
            self._intertwine and self._translation(0.5 * fraction / n)
            self._bath(fraction / n)
            self._num_rattles > 0 and self.addConstrainVelocities()
            self._intertwine and self._translation(0.5 * fraction / n)
            if n > 1:
                self.addComputeGlobal("ibath", "ibath + 1")
                self.endBlock()
            self._intertwine or self._translation(0.5 * fraction)

    def _integrate_respa_unrolled(self, fraction, scale):
        if scale >= 0:
            n = self._respa_loops[scale]
            for i in range(n):
                self._boost(
                    fraction / (2 * n if self._scheme == "VV-Middle" and i == 0 else n), scale
                )
                self._integrate_respa_unrolled(fraction / n, scale - 1)
                self._scheme == "VV-Middle" and i == n - 1 and self._boost(
                    fraction / (2 * n), scale
                )
        else:
            n = self._bath_loops
            self._intertwine or self._translation(0.5 * fraction)
            for i in range(n):
                self._intertwine and self._translation(fraction / (2 * n if i == 0 else n))
                self._bath(fraction / n)
                self._num_rattles > 0 and self.addConstrainVelocities()
                i == n - 1 and self._intertwine and self._translation(fraction / (2 * n))
            self._intertwine or self._translation(0.5 * fraction)

    def _translation(self, fraction):
        if self._num_rattles > 1:
            self.addComputeGlobal("irattle", "0")
            self.beginWhileBlock(f"irattle < {self._num_rattles-1/2}")
        self.addComputePerDof("x", f"x + {fraction/max(1, self._num_rattles)}*dt*v")
        if self._num_rattles > 0:
            self.addComputePerDof("x0", "x")
            self.addConstrainPositions()
            self.addComputePerDof("v", f"v + (x - x0)/({fraction/self._num_rattles}*dt)")
            self.addConstrainVelocities()
        if self._num_rattles > 1:
            self.addComputeGlobal("irattle", "irattle + 1")
            self.endBlock()

    def _boost(self, fraction, scale):
        if len(self._respa_loops) > 1:
            if scale - 1 in self._subtractive_groups:
                self.addComputePerDof("f_emb", f"f{scale-1}")
                self.addComputePerDof("v", f"v + {fraction}*dt*(f{scale}-f_emb)/m")
            else:
                self.addComputePerDof("v", f"v + {fraction}*dt*f{scale}/m")
        else:
            self.addComputePerDof("v", f"v + {fraction}*dt*f/m")
        self._num_rattles > 0 and self.addConstrainVelocities()

    def _bath(self, fraction):
        return


class GeodesicLangevinIntegrator(AbstractMiddleRespaIntegrator):
    """
    A geodesic Langevin integrator :cite:`Leimkuhler_2016`, which can be integrated by using
    either the LF-Middle or the VV-Middle scheme :cite:`Zhang_2019`.

    .. note:
        The VV-Middle scheme is also known as the BAOAB :cite:`Leimkuhler_2016` method.

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
        num_rattles : int, default=1
            The number of RATTLE computations for geodesic integration :cite:`Leimkuhler_2016`.
            If ``num_rattles=0``, then no constraints are considered at all.
        scheme : str, default='LF-Middle'
            Which splitting scheme will be used. Valid options are 'VV-Middle' and 'LF-Middle'.
        **kwargs
            All other keyword arguments in :class:`AbstractMiddleRespaIntegrator`.

    Example
    -------
        >>> import ufedmm
        >>> dt = 2*unit.femtoseconds
        >>> temp = 300*unit.kelvin
        >>> gamma = 10/unit.picoseconds
        >>> ufedmm.GeodesicLangevinIntegrator(temp, gamma, dt, num_rattles=1, scheme='VV-Middle')
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
           8: v <- z*v + sqrt((1 - z*z)*kT/m)*gaussian; z = exp(-friction*1.0*dt)
           9: constrain velocities
          10: x <- x + 0.5*dt*v
          11: x0 <- x
          12: constrain positions
          13: v <- v + (x - x0)/(0.5*dt)
          14: constrain velocities
          15: v <- v + 0.5*dt*f/m
          16: constrain velocities

    """

    def __init__(
        self,
        temperature,
        friction_coefficient,
        step_size,
        num_rattles=1,
        scheme="LF-Middle",
        **kwargs,
    ):
        super().__init__(temperature, step_size, num_rattles=num_rattles, scheme=scheme, **kwargs)
        self.addGlobalVariable("friction", friction_coefficient)

    def _bath(self, fraction):
        expression = f"z*v + sqrt((1 - z*z)*kT/m)*gaussian; z = exp(-friction*{fraction}*dt)"
        self.addComputePerDof("v", expression)


class MiddleMassiveNHCIntegrator(AbstractMiddleRespaIntegrator):
    """
    A massive, middle-type Nose-Hoover Chain Thermostat solver :cite:`Martyna_1992`
    with optional multiple time-scale integration via RESPA.

    To enable RESPA, the forces in OpenMM system must be split into distinct force
    groups and the keyword ``respa_loop`` (see below) must be a list with multiple entries.

    Parameters
    ----------
        temperature : float or unit.Quantity
            The temperature.
        time_constant : float or unit.Quantity
            The characteristic time constant.
        step_size : float or unit.Quantity
            The time-step size.

    Keyword Args
    ------------
        nchain : int, default=2
            The number of thermostats in each Nose-Hoover chain.
        track_energy : bool, default=False
            Whether to track the thermostat energy term.
        **kwargs
            All keyword arguments in :class:`AbstractMiddleRespaIntegrator`, except ``num_rattles``.

    Example
    -------
        >>> import ufedmm
        >>> temp, tau, dt = 300*unit.kelvin, 10*unit.femtoseconds, 2*unit.femtoseconds
        >>> integrator = ufedmm.MiddleMassiveNHCIntegrator(
        ...     temp, tau, dt, respa_loops=[4, 1], unroll_loops=False
        ... )
        >>> print(integrator)
        Per-dof variables:
          kT, Q, v1, v2
        Global variables:
          irespa0 = 0.0
        Computation steps:
           0: allow forces to update the context state
           1: v <- v + 0.5*dt*f1/m
           2: irespa0 <- 0
           3: while (irespa0 < 3.5):
           4:    v <- v + 0.125*dt*f0/m
           5:    x <- x + 0.125*dt*v
           6:    v2 <- v2 + 0.125*dt*(Q*v1^2 - kT)/Q
           7:    v1 <- (v1*z + 0.125*dt*(m*v^2 - kT)/Q)*z; z=exp(-0.0625*dt*v2)
           8:    v <- v*exp(-0.25*dt*v1)
           9:    v1 <- (v1*z + 0.125*dt*(m*v^2 - kT)/Q)*z; z=exp(-0.0625*dt*v2)
          10:    v2 <- v2 + 0.125*dt*(Q*v1^2 - kT)/Q
          11:    x <- x + 0.125*dt*v
          12:    v <- v + 0.125*dt*f0/m
          13:    irespa0 <- irespa0 + 1
          14: end
          15: v <- v + 0.5*dt*f1/m

    """

    def __init__(
        self, temperature, time_constant, step_size, nchain=2, track_energy=False, **kwargs
    ):
        if "num_rattles" in kwargs.keys() and kwargs["num_rattles"] != 0:
            raise ValueError(f"{self.__class__.__name__} cannot handle constraints")
        self._tau = _standardized(time_constant)
        self._nchain = nchain
        self._track_energy = track_energy
        super().__init__(temperature, step_size, **kwargs)
        self.addPerDofVariable("Q", 0)
        for i in range(nchain):
            self.addPerDofVariable(f"v{i+1}", 0)
            if track_energy:
                self.addPerDofVariable(f"eta{i+1}", 0)

    def update_temperatures(self, system_temperature, extended_space_temperatures):
        super().update_temperatures(system_temperature, extended_space_temperatures)
        Q = [self._tau**2 * kT for kT in self.getPerDofVariableByName("kT")]
        self.setPerDofVariableByName("Q", Q)

    def _bath(self, fraction):
        n = self._nchain

        def a(i):
            return f"(Q*v{i-1}^2 - kT)/Q" if i > 1 else "(m*v^2 - kT)/Q"

        def z(i):
            return f"exp(-{fraction/4}*dt*v{i+1})"

        self.addComputePerDof(f"v{n}", f"v{n} + {fraction/2}*dt*{a(n)}")
        for i in reversed(range(1, n)):
            self.addComputePerDof(f"v{i}", f"(v{i}*z + {fraction/2}*dt*{a(i)})*z; z={z(i)}")
        self.addComputePerDof("v", f"v*exp(-{fraction}*dt*v1)")
        for i in range(1, n):
            self.addComputePerDof(f"v{i}", f"(v{i}*z + {fraction/2}*dt*{a(i)})*z; z={z(i)}")
        self.addComputePerDof(f"v{n}", f"v{n} + {fraction/2}*dt*{a(n)}")


class MiddleMassiveGGMTIntegrator(AbstractMiddleRespaIntegrator):
    """
    A massive, middle-type Generalized Gaussian Moment Thermostat :cite:`Liu_2000`
    solver with optional multiple time-scale integration via RESPA.

    To enable RESPA, the forces in OpenMM system must be split into distinct force
    groups and the keyword ``respa_loop`` (see below) must be a list with multiple entries.

    Parameters
    ----------
        temperature : float or unit.Quantity
            The temperature.
        time_constant : float or unit.Quantity
            The characteristic time constant.
        step_size : float or unit.Quantity
            The time-step size.

    Keyword Args
    ------------
        **kwargs
            All keyword arguments in :class:`AbstractMiddleRespaIntegrator`, except ``num_rattles``.

    Example
    -------
        >>> import ufedmm
        >>> temp, tau, dt = 300*unit.kelvin, 10*unit.femtoseconds, 2*unit.femtoseconds
        >>> integrator = ufedmm.MiddleMassiveGGMTIntegrator(temp, tau, dt)
        >>> print(integrator)
        Per-dof variables:
          kT, Q1, Q2, v1, v2
        Computation steps:
           0: allow forces to update the context state
           1: v <- v + 0.5*dt*f/m
           2: x <- x + 0.5*dt*v
           3: v1 <- v1 + 0.5*dt*(m*v^2 - kT)/Q1
           4: v2 <- v2 + 0.5*dt*((m*v^2)^2/3 - kT^2)/Q2
           5: v <- v*exp(-1.0*dt*(v1 + kT*v2))/sqrt(1 + 2.0*dt*m*v^2*v2/3)
           6: v1 <- v1 + 0.5*dt*(m*v^2 - kT)/Q1
           7: v2 <- v2 + 0.5*dt*((m*v^2)^2/3 - kT^2)/Q2
           8: x <- x + 0.5*dt*v
           9: v <- v + 0.5*dt*f/m

    """

    def __init__(self, temperature, time_constant, step_size, **kwargs):
        if "num_rattles" in kwargs.keys() and kwargs["num_rattles"] != 0:
            raise ValueError(f"{self.__class__.__name__} cannot handle constraints")
        self._tau = _standardized(time_constant)
        super().__init__(temperature, step_size, **kwargs)
        self.addPerDofVariable("Q1", 0)
        self.addPerDofVariable("Q2", 0)
        self.addPerDofVariable("v1", 0)
        self.addPerDofVariable("v2", 0)

    def set_extended_space_time_constants(self, time_constants):
        self._xs_taus = [_standardized(tau) for tau in time_constants]

    def update_temperatures(self, system_temperature, extended_space_temperatures):
        super().update_temperatures(system_temperature, extended_space_temperatures)
        kT_vectors = self.getPerDofVariableByName("kT")
        kT3_vectors = [openmm.Vec3(kT.x**3, kT.y**3, kT.z**3) for kT in kT_vectors]
        if hasattr(self, "_xs_taus"):
            num_particles = len(kT_vectors) - len(extended_space_temperatures)
            taus = [self._tau] * num_particles + self._xs_taus
            Q1 = [kT * tau**2 for kT, tau in zip(kT_vectors, taus)]
            Q2 = [8 / 3 * kT3 * tau**2 for kT3, tau in zip(kT3_vectors, taus)]
        else:
            Q1 = [kT * self._tau**2 for kT in kT_vectors]
            Q2 = [8 / 3 * kT3 * self._tau**2 for kT3 in kT3_vectors]
        self.setPerDofVariableByName("Q1", Q1)
        self.setPerDofVariableByName("Q2", Q2)

    def _bath(self, fraction):
        self.addComputePerDof("v1", f"v1 + {fraction/2}*dt*(m*v^2 - kT)/Q1")
        self.addComputePerDof("v2", f"v2 + {fraction/2}*dt*((m*v^2)^2/3 - kT^2)/Q2")
        self.addComputePerDof(
            "v", f"v*exp(-{fraction}*dt*(v1 + kT*v2))/sqrt(1 + {2*fraction}*dt*m*v^2*v2/3)"
        )
        self.addComputePerDof("v1", f"v1 + {fraction/2}*dt*(m*v^2 - kT)/Q1")
        self.addComputePerDof("v2", f"v2 + {fraction/2}*dt*((m*v^2)^2/3 - kT^2)/Q2")


class RegulatedNHLIntegrator(AbstractMiddleRespaIntegrator):
    """
    A regulated version of the massive Nose-Hoover-Langevin :cite:`Samoletov_2007,Leimkuhler_2009`
    method. Regulation means that the system Hamiltonian is modified so that velocities remain
    below a temperature-dependent limit. This is closely related to the SIN(R) method
    :cite:`Leimkuhler_2013` and allows multiple time-scale integration with very large outer time
    steps, without resonance.

    .. info:
        If `regulation_parameter = 1` (default), this method is equivalent to SIN(R) with a single
        thermostat per degree of freedom (that is, `L=1`).

    The following :term:`SDE` system is solved for every degree of freedom in the system:

    .. math::
        & dr_i = v_i dt \\\\
        & dp_i = F_i dt - v_{\\eta_i} m_i v_i dt \\\\
        & dv_{\\eta_i} = \\frac{1}{Q}\\left(\\frac{n+1}{n} m_i v_i^2 - k_B T\\right) dt
                - \\gamma v_{\\eta_i} dt + \\sqrt{\\frac{2\\gamma k_B T}{Q}} dW_i,

    where:

    .. math::
        v_i = c_i \\tanh\\left(\\frac{p_i}{m_i c_i}\\right).

    Here, :math:`n` is the regulation parameter and :math:`c_i = \\sqrt{\\frac{n k T}{m_i}}` is
    the maximum speed for degree of freedom `i`. The inertial parameter :math:`Q` is defined as
    :math:`Q = n k_B T \\tau^2`, with :math:`\\tau` being a relaxation time :cite:`Tuckerman_1992`.
    An approximate solution is obtained by applying the Trotter-Suzuki splitting formula:

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
                    \\sinh\\left(\\frac{p_i^0}{m_i c_i}\\right) e^{- v_{\\eta_i} t}
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
        regulation_parameter : int or float
            The regulation parameter n.

    Keyword Args
    ------------
        semi_regulated : bool, default=True
            Whether to use the semi-regulated NHL version of the method instead of its
            fully-regulated version.
        split_ornstein_uhlenbeck : bool, default=True
            Whether to split the drifted Ornstein-Uhlenbeck operator.
        **kwargs
            All keyword arguments in :class:`AbstractMiddleRespaIntegrator`, except ``num_rattles``.

    """

    def __init__(
        self,
        temperature,
        time_constant,
        friction_coefficient,
        step_size,
        regulation_parameter,
        semi_regulated=True,
        split_ornstein_uhlenbeck=True,
        **kwargs,
    ):
        if "num_rattles" in kwargs.keys() and kwargs["num_rattles"] != 0:
            raise ValueError(f"{self.__class__.__name__} cannot handle constraints")
        self._tau = np.sqrt(regulation_parameter) * time_constant
        self._n = regulation_parameter
        self._split = split_ornstein_uhlenbeck
        self._semi_regulated = semi_regulated
        super().__init__(temperature, step_size, **kwargs)
        self.addPerDofVariable("invQ", 0)
        self.addPerDofVariable("v_eta", 0)
        self.addPerDofVariable("c", 0)
        self.addGlobalVariable("friction", friction_coefficient)
        self.addGlobalVariable("omega", 1.0 / self._tau)
        self.addGlobalVariable("aa", 0)
        self.addGlobalVariable("bb", 0)

    def update_temperatures(self, system_temperature, extended_space_temperatures):
        super().update_temperatures(system_temperature, extended_space_temperatures)
        kT_vectors = self.getPerDofVariableByName("kT")
        tauSq = _standardized(self._tau) ** 2
        Q = [tauSq * kT for kT in kT_vectors]
        invQ = [openmm.Vec3(*map(lambda x: 1 / x if x > 0.0 else 0.0, q)) for q in Q]
        self.setPerDofVariableByName("invQ", invQ)

    def _step_initialization(self):
        self.addComputePerDof("c", f"sqrt({self._n}*kT/m)")
        n = np.prod(self._respa_loops) * self._bath_loops
        self.addComputeGlobal("aa", f"exp(-friction*dt/{n})")
        self.addComputeGlobal("bb", "omega*sqrt(1-aa^2)")

    def _translation(self, fraction):
        n = self._n
        if self._semi_regulated:
            expression = f"0.5*m*v*c*tanh(v/c); c=sqrt({n}*kT/m)"
        else:
            expression = f"{0.5*(n+1)/n}*m*(c*tanh(v/c))^2; c=sqrt({n}*kT/m)"
        self.setKineticEnergyExpression(expression)
        self.addComputePerDof("x", f"x + c*tanh(v/c)*{fraction}*dt")

    def _bath(self, fraction):
        n = self._n

        if self._semi_regulated:
            G = "; G=(m*v*c*tanh(v/c) - kT)*invQ"
        else:
            G = f"; G=({(n+1)/n}*m*(c*tanh(v/c))^2 - kT)*invQ"

        if self._split:
            boost = f"v_eta + G*{0.5*fraction}*dt" + G

        if self._semi_regulated:
            scaling = f"v*exp(-v_eta*{0.5*fraction}*dt)"
        else:
            scaling = "c*asinh_z"
            scaling += (
                "; asinh_z=(2*step(z)-1)*log(select(step(za-1E8),2*za,za+sqrt(1+z*z))); za=abs(z)"
            )
            scaling += f"; z=sinh(v/c)*exp(-v_eta*{0.5*fraction}*dt)"

        if self._split:
            Ornstein_Uhlenbeck = "v_eta*aa + bb*gaussian"
        else:
            Ornstein_Uhlenbeck = "v_eta*aa + G*(1-aa)/friction + bb*gaussian" + G

        self._split and self.addComputePerDof("v_eta", boost)
        self.addComputePerDof("v", scaling)
        self.addComputePerDof("v_eta", Ornstein_Uhlenbeck)
        self.addComputePerDof("v", scaling)
        self._split and self.addComputePerDof("v_eta", boost)
