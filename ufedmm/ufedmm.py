"""
.. module:: ufedmm
   :platform: Unix, Windows
   :synopsis: Unified Free Energy Dynamics with OpenMM

.. moduleauthor:: Charlles Abreu <abreu@eq.ufrj.br>

.. _Context:
    http://docs.openmm.org/latest/api-python/generated/openmm.openmm.Context.html
.. _CustomCVForce:
    http://docs.openmm.org/latest/api-python/generated/openmm.openmm.CustomCVForce.html
.. _CustomIntegrator:
    http://docs.openmm.org/latest/api-python/generated/openmm.openmm.CustomIntegrator.html
.. _Force:
    http://docs.openmm.org/latest/api-python/generated/openmm.openmm.Force.html
.. _Integrator:
    http://docs.openmm.org/latest/api-python/generated/openmm.openmm.Integrator.html
.. _Platform:
    http://docs.openmm.org/latest/api-python/generated/openmm.openmm.Platform.html
.. _System:
    http://docs.openmm.org/latest/api-python/generated/openmm.openmm.System.html
.. _State:
    http://docs.openmm.org/latest/api-python/generated/openmm.openmm.State.html

"""

import functools
from copy import deepcopy

import numpy as np
import openmm
from openmm import app, unit

import ufedmm


def _standardized(quantity):
    """
    Returns the numerical value of a quantity in a unit of measurement compatible with the
    Molecular Dynamics unit system (mass in Da, distance in nm, time in ps, temperature in K,
    energy in kJ/mol, angle in rad).

    """
    if unit.is_quantity(quantity):
        return quantity.value_in_unit_system(unit.md_unit_system)
    else:
        return quantity


def _update_RMSD_forces(system):
    N = system.getNumParticles()

    def update_RMSDForce(force):
        positions = force.getReferencePositions()._value
        if len(positions) >= N:
            positions = positions[:N]
        else:
            positions += [openmm.Vec3(0, 0, 0)] * (N - len(positions))
        force.setReferencePositions(positions)

    for force in system.getForces():
        if isinstance(force, openmm.RMSDForce):
            update_RMSDForce(force)
        elif isinstance(force, openmm.CustomCVForce):
            for index in range(force.getNumCollectiveVariables()):
                cv = force.getCollectiveVariable(index)
                if isinstance(cv, openmm.RMSDForce):
                    update_RMSDForce(cv)


class CollectiveVariable(object):
    """
    A function of the particle coordinates, evaluated by means of an OpenMM Force_ object.

    Quoting OpenMM's CustomCVForce_ manual entry:

        "Each collective variable is defined by a Force object. The Force's potential energy is
        computed, and that becomes the value of the variable. This provides enormous flexibility
        in defining collective variables, especially by using custom forces. Anything that can
        be computed as a potential function can also be used as a collective variable."

    Parameters
    ----------
        id : str
            A valid identifier string for this collective variable.
        force : openmm.Force
            An OpenMM Force_ object whose energy function is used to evaluate this collective
            variable.

    Example
    -------
        >>> import openmm
        >>> import ufedmm
        >>> from openmm import unit
        >>> cv = ufedmm.CollectiveVariable('psi', openmm.CustomTorsionForce('theta'))
        >>> cv.force.addTorsion(0, 1, 2, 3, [])
        0

    """

    def __init__(self, id, force):
        if not id.isidentifier():
            raise ValueError("Parameter id must be a valid variable identifier")
        self.id = id
        self.force = force

    def _create_context(self, system, positions):
        system_copy = deepcopy(system)
        for force in system_copy.getForces():
            force.setForceGroup(0)
        force_copy = deepcopy(self.force)
        force_copy.setForceGroup(1)
        system_copy.addForce(force_copy)
        platform = openmm.Platform.getPlatformByName("Reference")
        _update_RMSD_forces(system_copy)
        context = openmm.Context(system_copy, openmm.CustomIntegrator(0), platform)
        context.setPositions(positions)
        return context

    def evaluate(self, system, positions, cv_unit=None):
        """
        Computes the value of the collective variable for a given system and a given set of particle
        coordinates.

        Parameters
        ----------
            system : openmm.System
                The system for which the collective variable will be evaluated.
            positions : list of openmm.Vec3
                A list whose size equals the number of particles in the system and which contains
                the coordinates of these particles.

        Keyword Args
        ------------
            cv_unit : unit.Unit, default=None
                The unity of measurement of the collective variable. If this is `None`, then a
                numerical value is returned based on the OpenMM default units.

        Returns
        -------
            float or unit.Quantity

        Example
        -------
            >>> import ufedmm
            >>> from openmm import unit
            >>> model = ufedmm.AlanineDipeptideModel()
            >>> model.phi.evaluate(model.system, model.positions)
            3.141592653589793
            >>> model.psi.evaluate(model.system, model.positions)
            3.141592653589793

        """
        context = self._create_context(system, positions)
        energy = context.getState(getEnergy=True, groups={1}).getPotentialEnergy()
        value = energy.value_in_unit(unit.kilojoules_per_mole)
        if cv_unit is not None:
            value *= cv_unit / _standardized(1 * cv_unit)
        return value

    def effective_mass(self, system, positions, cv_unit=None):
        """
        Computes the effective mass of the collective variable for a given system and a given set of
        particle coordinates.

        The effective mass of a collective variable :math:`q(\\mathbf{r})` is defined as
        :cite:`Cuendet_2014`:

        .. math::
            m_\\mathrm{eff} = \\left(
                \\sum_{j=1}^N \\frac{1}{m_j} \\left\\|\\frac{dq}{d\\mathbf{r}_j}\\right\\|^2
            \\right)^{-1}

        Parameters
        ----------
            system : openmm.System
                The system for which the collective variable will be evaluated.
            positions : list of openmm.Vec3
                A list whose size equals the number of particles in the system and which contains
                the coordinates of these particles.

        Keyword Args
        ------------
            cv_unit : unit.Unit, default=None
                The unity of measurement of the collective variable. If this is `None`, then a
                numerical value is returned based on the OpenMM default units.

        Returns
        -------
            float or unit.Quantity

        Example
        -------
            >>> import ufedmm
            >>> from openmm import unit
            >>> model = ufedmm.AlanineDipeptideModel()
            >>> model.phi.effective_mass(model.system, model.positions)  # doctest: +ELLIPSIS
            0.04795887...
            >>> model.psi.effective_mass(model.system, model.positions)  # doctest: +ELLIPSIS
            0.05115582...

        """
        context = self._create_context(system, positions)
        forces = _standardized(context.getState(getForces=True, groups={1}).getForces(asNumpy=True))
        denom = sum(
            f.dot(f) / _standardized(system.getParticleMass(i)) for i, f in enumerate(forces)
        )
        effective_mass = 1.0 / float(denom)
        if cv_unit is not None:
            factor = _standardized(1 * cv_unit) ** 2
            effective_mass *= factor * unit.dalton * (unit.nanometers / cv_unit) ** 2
        return effective_mass


class DynamicalVariable(object):
    """
    An extended phase-space variable, whose dynamics is coupled to that of one of more collective
    variables of a system.

    The coupling occurs in the form of a potential energy term involving this dynamical variable
    and its associated collective variables.

    The default potential is a harmonic driving of the type:

    .. math::
        V(s, \\mathbf r) = \\frac{\\kappa}{2} [s - q(\\mathbf r)]^2

    where :math:`s` is the new dynamical variable, :math:`q(\\mathbf r)` is its associated
    collective variable, and :math:`kappa` is a force constant.

    Parameters
    ----------
        id : str
            A valid identifier string for this dynamical variable.
        min_value : float or unit.Quantity
            The minimum allowable value for this dynamical variable.
        max_value : float or unit.Quantity
            The maximum allowable value for this dynamical variable.
        mass : float or unit.Quantity
            The mass assigned to this dynamical variable, whose unit of measurement must be
            compatible with `unit.dalton*(unit.nanometers/X)**2`, where `X` is the unit of
            measurement of the dynamical variable itself.
        temperature : float or unit.Quantity
            The temperature of the heat bath attached to this variable.
        colvars : :class:`~ufedmm.ufedmm.CollectiveVariable` or list thereof
            Either a single colective variable or a list.
        potential : float or unit.Quantity or str
            Either the value of the force constant of a harmonic driving potential or an algebraic
            expression giving the energy of the system as a function of this dynamical variable and
            its associated collective variable. Such expression can also contain a set of global
            parameters, whose values must be passed as keyword arguments (see below).

    Keyword Args
    ------------
        periodic : bool, default=True
            Whether the collective variable is periodic with period `L=max_value-min_value`.
        sigma : float or unit.Quantity, default=None
            The standard deviation. If this is `None`, then no bias will be considered.
        grid_size : int, default=None
            The grid size. If this is `None` and `sigma` is finite, then a convenient value will be
            automatically chosen.
        **parameters
            Names and values of global parameters present in the algebraic expression defined as
            `potential` (see above).

    Example
    -------
        >>> import openmm
        >>> import ufedmm
        >>> from openmm import unit
        >>> cv = ufedmm.CollectiveVariable('psi', openmm.CustomTorsionForce('theta'))
        >>> cv.force.addTorsion(0, 1, 2, 3, [])
        0
        >>> mass = 50*unit.dalton*(unit.nanometer/unit.radians)**2
        >>> K = 1000*unit.kilojoules_per_mole/unit.radians**2
        >>> Ts = 1500*unit.kelvin
        >>> ufedmm.DynamicalVariable('s_psi', -180*unit.degrees, 180*unit.degrees, mass, Ts, cv, K)
        <s_psi in [-3.141592653589793, 3.141592653589793], periodic, m=50, T=1500>

    """

    def __init__(
        self,
        id,
        min_value,
        max_value,
        mass,
        temperature,
        colvars,
        potential,
        periodic=True,
        sigma=None,
        grid_size=None,
        **parameters,
    ):
        self.id = id
        self.min_value = _standardized(min_value)
        self.max_value = _standardized(max_value)
        self._range = self.max_value - self.min_value
        self.mass = _standardized(mass)
        self.temperature = _standardized(temperature)

        self.colvars = colvars if isinstance(colvars, (list, tuple)) else [colvars]

        if isinstance(potential, str):
            self.potential = potential
            self.parameters = {key: _standardized(value) for key, value in parameters.items()}
        else:
            cv_id = self.colvars[0].id
            if periodic:
                self.potential = f"0.5*K_{cv_id}*min(d{cv_id},{self._range}-d{cv_id})^2"
                self.potential += f"; d{cv_id}=abs({cv_id}-{self.id})"
            else:
                self.potential = f"0.5*K_{cv_id}*({cv_id}-{self.id})^2"
            self.parameters = {f"K_{cv_id}": _standardized(potential)}

        self.periodic = periodic

        if sigma is None or sigma == 0.0:
            self.sigma = self.grid_size = None
        else:
            self.sigma = _standardized(sigma)
            self._scaled_variance = (self.sigma / self._range) ** 2
            if grid_size is None:
                self.grid_size = int(np.ceil(5 * self._range / self.sigma)) + 1
            else:
                self.grid_size = grid_size

        self.force = openmm.CustomExternalForce(self._get_energy_function())
        self.force.addGlobalParameter("Lx", 0.0)
        self.force.addParticle(0, [])

    def __repr__(self):
        properties = f"m={self.mass}, T={self.temperature}"
        status = "periodic" if self.periodic else "non-periodic"
        return f"<{self.id} in [{self.min_value}, {self.max_value}], {status}, {properties}>"

    def __getstate__(self):
        return dict(
            id=self.id,
            min_value=self.min_value,
            max_value=self.max_value,
            mass=self.mass,
            temperature=self.temperature,
            colvars=self.colvars,
            potential=self.potential,
            periodic=self.periodic,
            sigma=self.sigma,
            grid_size=self.grid_size,
            **self.parameters,
        )

    def __setstate__(self, kw):
        self.__init__(**kw)

    def _particle_mass(self, Lx):
        length = Lx if self.periodic else Lx / 2
        return self.mass * (self._range / length) ** 2

    def _particle_position(self, value, Lx, y=0):
        length = Lx if self.periodic else Lx / 2
        return openmm.Vec3(length * (value - self.min_value) / self._range, y, 0) * unit.nanometer

    def _get_energy_function(self, index=""):
        """
        Returns the algebraic expression that transforms the x coordinate of a particle into this
        dynamical variables.

        Keyword Args
        ------------
            index : str or int, default=''
                An index for the particle in question, if needed.

        Returns
        -------
            str

        """
        if self.periodic:
            energy = f"{self.min_value}+{self._range}*(x{index}/Lx-floor(x{index}/Lx))"
        else:
            energy = f"{self.min_value}+{2*self._range}*min(pos{index},1-pos{index})"
            energy += f";pos{index}=x{index}/Lx-floor(x{index}/Lx)"
        return energy

    def evaluate(self, x, Lx):
        """
        Computes the value of this dynamical variable for a given x coordinate and a given length of
        the simulation box in the x direction.

        Parameters
        ----------
            x : float or unit.Quantity
                The x coordinate.
            Lx : float or unit.Quantity
                The length of the simulation box in the x direction.

        Returns
        -------
            float

        """
        pos = x / Lx - np.floor(x / Lx)
        if self.periodic:
            return self.min_value + self._range * pos
        else:
            return self.min_value + 2 * self._range * min(pos, 1 - pos)


def _get_energy_function(variables):
    energies = [v.potential.split(";", 1) for v in variables]
    energy_terms = [energy[0] for energy in energies]
    definitions = [energy[1] for energy in energies if len(energy) == 2]
    expression = ";".join(["+".join(energy_terms)] + list(definitions))
    return expression


def _get_parameters(variables):
    parameters = {}
    for v in variables:
        parameters.update(v.parameters)
    return parameters


class PeriodicTask(object):
    def __init__(self, frequency):
        self.frequency = frequency

    def initialize(self, simulation, force_group):
        pass

    def update(self, simulation, steps):
        pass

    def describeNextReport(self, simulation):
        steps = self.frequency - simulation.context.getStepCount() % self.frequency
        return (steps, True, False, False, False, False)

    def report(self, simulation, state):
        pass

    def saveCheckpoint(self, file):
        pass

    def loadCheckpoint(self, file):
        pass


class _Metadynamics(PeriodicTask):
    """
    Extended-space Metadynamics.

    Parameters
    ----------
        variables : list of :class:`DynamicalVariable`
            A list of extended-space dynamical variables to which the metadynamics bias must be
            applied. In fact, dynamical variables with `sigma = None` will not be considered.
        height : float or unit.Quantity
            The height of the Gaussian potential hills to be deposited. If the `bias_factor` keyword
            is defined (see below), then this is the unscaled height.
        frequency : int
            The frequency of Gaussian hill deposition.

    Keyword Args
    ------------
        bias_factor : float, default=None
            Scales the height of the hills added to the bias potential. If it is `None`, then the
            hills will have a constant height over time. For a bias factor to be applicable, all
            bias variables must be at the same temperature T. The extended-space dynamical
            variables are sampled as if the effective temperature were T*bias_factor.
        buffer_size : int, default=100
            The buffer size.
        grid_expansion : int, default=20
            The number of extra grid points to be used in periodic directions of multidimensional
            tabulated functions. This aims at avoiding boundary discontinuity artifacts.
        enforce_gridless : bool, default=False
            Enforce gridless metadynamics even for 1D to 3D problems.

    """

    def __init__(
        self,
        variables,
        height,
        frequency,
        bias_factor=None,
        buffer_size=100,
        grid_expansion=20,
        enforce_gridless=False,
    ):
        super().__init__(frequency)
        self.bias_indices = [i for i, v in enumerate(variables) if v.sigma is not None]
        self.bias_variables = [variables[i] for i in self.bias_indices]
        self.initial_height = self.height = _standardized(height)
        self.bias_factor = bias_factor
        if bias_factor is not None:
            temperature = self.bias_variables[0].temperature
            if any(v.temperature != temperature for v in self.bias_variables):
                raise ValueError(
                    "Well-tempered metadynamics requires all variables at the same temperature"
                )
            self.delta_kT = (
                (bias_factor - 1) * unit.MOLAR_GAS_CONSTANT_R * temperature * unit.kelvin
            )
        self.buffer_size = buffer_size
        self.grid_expansion = grid_expansion
        self._use_grid = len(self.bias_variables) < 4 and not enforce_gridless
        self.force = self._interpolation_grid_force() if self._use_grid else self._hills_force()

    def _interpolation_grid_force(self):
        self._widths = []
        self._bounds = []
        self._extra_points = []
        full_periodic = self._full_periodic = all(v.periodic for v in self.bias_variables)
        for v in self.bias_variables:
            extra_points = (
                min(self.grid_expansion, v.grid_size - 1) if v.periodic and not full_periodic else 0
            )
            extra_range = extra_points * v._range / (v.grid_size - 1)
            self._widths.append(v.grid_size + 2 * extra_points)
            self._bounds += [v.min_value - extra_range, v.max_value + extra_range]
            self._extra_points.append(extra_points)
        self._bias = np.zeros(np.prod(self._widths))
        num_bias_variables = len(self.bias_variables)
        if num_bias_variables == 1:
            self._table = openmm.Continuous1DFunction(self._bias, *self._bounds, full_periodic)
        elif num_bias_variables == 2:
            self._table = openmm.Continuous2DFunction(
                *self._widths, self._bias, *self._bounds, full_periodic
            )
        else:
            self._table = openmm.Continuous3DFunction(
                *self._widths, self._bias, *self._bounds, full_periodic
            )
        expression = f'bias({",".join(v.id for v in self.bias_variables)})'
        for i, v in enumerate(self.bias_variables):
            expression += f";{v.id}={v._get_energy_function(i+1)}"
        force = openmm.CustomCVForce(expression)
        for i in range(num_bias_variables):
            x = openmm.CustomExternalForce("x")
            x.addParticle(0, [])
            force.addCollectiveVariable(f"x{i+1}", x)
        force.addTabulatedFunction("bias", self._table)
        force.addGlobalParameter("Lx", 0)
        return force

    def _hills_force(self):
        num_bias_variables = len(self.bias_variables)
        centers = [f"center{i+1}" for i in range(num_bias_variables)]
        exponents = []
        for v, center in zip(self.bias_variables, centers):
            if v.periodic:  # von Mises
                factor = 2 * np.pi / v._range
                exponents.append(f"{1.0/(factor*v.sigma)**2}*(cos({factor}*({v.id}-{center}))-1)")
            else:  # Gauss
                exponents.append(f"({-0.5/v.sigma**2})*({v.id}-{center})^2")
        expression = f'height*exp({"+".join(exponents)})'
        for i, v in enumerate(self.bias_variables):
            expression += f";{v.id}={v._get_energy_function(i+1)}"
        force = openmm.CustomCompoundBondForce(num_bias_variables, expression)
        force.addPerBondParameter("height")
        for center in centers:
            force.addPerBondParameter(center)
        force.addGlobalParameter("Lx", 0)
        return force

    def _add_buffer(self, simulation):
        size = min(self.buffer_size, self._total_hills - self.force.getNumBonds())
        for i in range(size):
            self.force.addBond(self.particles, [0] * (len(self.bias_variables) + 1))
        simulation.context.reinitialize(preserveState=True)

    def add_bias(self, simulation, bias):
        if self._use_grid:
            self._bias += bias.ravel()
            if len(self.bias_variables) == 1:
                self._table.setFunctionParameters(self._bias, *self._bounds)
            else:
                self._table.setFunctionParameters(*self._widths, self._bias, *self._bounds)
            self.force.updateParametersInContext(simulation.context)

    def initialize(self, simulation):
        context = simulation.context
        np = context.getSystem().getNumParticles() - len(context.variables)
        self.particles = [np + index for index in self.bias_indices]
        self.Lx = context.getParameter("Lx")
        if self._use_grid:
            for i, particle in enumerate(self.particles):
                self.force.getCollectiveVariable(i).setParticleParameters(0, particle, [])
        else:
            self._num_hills = 0
            self._total_hills = 0
        simulation.system.addForce(self.force)
        context.reinitialize(preserveState=True)

    def update(self, simulation, steps):
        if self.bias_factor is not None:
            self.free_group = 31
            used_groups = set(f.getForceGroup() for f in simulation.system.getForces())
            while self.free_group in used_groups:
                self.free_group -= 1
            if self.free_group < 0:
                raise RuntimeError("Well-tempered Metadynamics requires free force groups")
        if not self._use_grid:
            steps_until_next_report = self.frequency - simulation.currentStep % self.frequency
            if steps_until_next_report > steps:
                required_hills = 0
            else:
                required_hills = (steps - steps_until_next_report) // self.frequency + 1
            self._total_hills = self._num_hills + required_hills
            self._add_buffer(simulation)

    def report(self, simulation, state):
        centers = state.getDynamicalVariables()
        if self.bias_factor is None:
            self.height = self.initial_height
        else:
            group = self.force.getForceGroup()
            self.force.setForceGroup(self.free_group)
            hills_state = simulation.context.getState(getEnergy=True, groups={self.free_group})
            self.force.setForceGroup(group)
            energy = hills_state.getPotentialEnergy()
            self.height = self.initial_height * np.exp(-energy / self.delta_kT)
        if self._use_grid:
            hills = []
            for i, v in enumerate(self.bias_variables):
                x = (centers[i] - v.min_value) / v._range
                dist = np.linspace(0, 1, num=v.grid_size) - x
                if v.periodic:  # von Mises
                    exponents = (np.cos(2 * np.pi * dist) - 1) / (
                        4 * np.pi * np.pi * v._scaled_variance
                    )
                    exponents[0] = exponents[-1] = 0.5 * (exponents[0] + exponents[-1])
                else:  # Gauss
                    exponents = -0.5 * dist * dist / v._scaled_variance
                hills.append(self.height * np.exp(exponents))
            ndim = len(self.bias_variables)
            bias = hills[0] if ndim == 1 else functools.reduce(np.multiply.outer, reversed(hills))
            if not self._full_periodic:
                for axis, v in enumerate(self.bias_variables):
                    if v.periodic:
                        n = self._extra_points[axis] + 1
                        begin = tuple(
                            slice(1, n) if i == axis else slice(None) for i in range(ndim)
                        )
                        end = tuple(
                            slice(-n, -1) if i == axis else slice(None) for i in range(ndim)
                        )
                        bias = np.concatenate((bias[end], bias, bias[begin]), axis=axis)
            self.add_bias(simulation, bias)
        else:
            if self._num_hills == self.force.getNumBonds():
                self._add_buffer(simulation)
            self.force.setBondParameters(self._num_hills, self.particles, [self.height] + centers)
            self._num_hills += 1
            self.force.updateParametersInContext(simulation.context)

    def saveCheckpoint(self, file):
        if self._use_grid:
            self._bias.tofile(file)
        else:
            np.array([self._num_hills], dtype=int).tofile(file)
            parameter_list = []
            for index in range(self._num_hills):
                _, parameters = self.force.getBondParameters(index)
                parameter_list.append(parameters)
            np.array(parameter_list).tofile(file)
        np.array([self.height]).tofile(file)

    def loadCheckpoint(self, file, context):
        if self._use_grid:
            self._bias = np.fromfile(file, count=len(self._bias))
            if len(self.bias_variables) == 1:
                self._table.setFunctionParameters(self._bias, *self._bounds)
            else:
                self._table.setFunctionParameters(*self._widths, self._bias, *self._bounds)
            self.force.updateParametersInContext(context)
        else:
            npars = len(self.bias_variables) + 1
            nhills = self._num_hills = np.fromfile(file, dtype=int, count=1)[0]
            parameter_array = np.fromfile(file, count=nhills * npars).reshape((nhills, npars))
            for index in range(nhills - self.force.getNumBonds()):
                self.force.addBond(self.particles, [0] * npars)
            self._total_hills = self.force.getNumBonds()
            for index, parameters in enumerate(parameter_array):
                self.force.setBondParameters(index, self.particles, parameters)
            for index in range(self._total_hills - nhills):
                self.force.setBondParameters(index, self.particles, [0] * npars)
            context.reinitialize(preserveState=True)
        self.height = np.fromfile(file, count=1)[0]


class ExtendedSpaceState(openmm.State):
    """
    An extension of OpenMM's State_ class.

    """

    def __init__(self, variables, state):
        self.__class__ = type(state.__class__.__name__, (self.__class__, state.__class__), {})
        self.__dict__ = state.__dict__
        self._variables = variables
        a, _, _ = self.getPeriodicBoxVectors()
        self._Lx = a.x

    def _split(self, vector, asNumpy=False):
        np = (vector.shape[0] if asNumpy else len(vector)) - len(self._variables)
        particles_contribution = vector[:np, :] if asNumpy else vector[:np]
        variables_contribution = vector[np:, 0] if asNumpy else [v.x for v in vector[np:]]
        return particles_contribution, variables_contribution

    def getPositions(self, asNumpy=False, extended=False):
        """
        Gets the positions of all physical particles and optionally also gets the positions
        of the extra particles from which the extended-space dynamical variables are computed.

        Keyword Args
        ------------
            asNumpy : bool, default=False
                Whether to return Numpy arrays instead of lists of openmm.Vec3.
            extended : bool, default=False
                Whether to include the positions of the extra particles from which the
                extended-space dynamical variables are computed.

        Returns
        -------
            list(openmm.Vec3)
                If `asNumpy=False` and `extended=False`.
            numpy.ndarray
                If `asNumpy=True` and `extended=False`.
            list(openmm.Vec3), list(float)
                If `asNumpy=False` and `extended=True`.
            numpy.ndarray, numpy.ndarray
                If `asNumpy=True` and `extended=True`.

        Raises
        ------
            Exception
                If positions were not requested in the ``context.getState()`` call.

        """
        positions, xvars = self._split(super().getPositions(asNumpy), asNumpy)
        return (positions, xvars) if extended else positions

    def getVelocities(self, asNumpy=False, extended=False):
        """
        Gets the velocities of all physical particles and optionally also gets the velocities
        of the extra particles associated to the extended-space dynamical variables.

        Keyword Args
        ------------
            asNumpy : bool, default=False
                Whether to return Numpy arrays instead of lists of openmm.Vec3.
            extended : bool, default=False
                Whether to include the velocities of the extra particles.

        Returns
        -------
            list(openmm.Vec3)
                If `asNumpy=False` and `extended=False`.
            numpy.ndarray
                If `asNumpy=True` and `extended=False`.
            list(openmm.Vec3), list(float)
                If `asNumpy=False` and `extended=True`.
            numpy.ndarray, numpy.ndarray
                If `asNumpy=True` and `extended=True`.

        Raises
        ------
            Exception
                If velocities were not requested in the ``context.getState()`` call.

        """
        velocities = super().getVelocities(asNumpy)
        if not extended:
            velocities, _ = self._split(velocities, asNumpy)
        return velocities

    def getDynamicalVariables(self):
        """
        Gets the values of the extended-space dynamical variables.

        Returns
        -------
            list(float)

        Raises
        ------
            Exception
                If positions were not requested in the ``context.getState()`` call.

        Example
        -------
            >>> import ufedmm
            >>> import numpy as np
            >>> model = ufedmm.AlanineDipeptideModel()
            >>> integrator = ufedmm.CustomIntegrator(300, 0.001)
            >>> args = [-np.pi, np.pi, 50, 1500, model.phi, 1000]
            >>> s_phi = ufedmm.DynamicalVariable('s_phi', *args)
            >>> s_psi = ufedmm.DynamicalVariable('s_psi', *args)
            >>> context = ufedmm.ExtendedSpaceContext([s_phi, s_psi], model.system, integrator)
            >>> context.setPositions(model.positions)
            >>> context.getState(getPositions=True).getDynamicalVariables()
            [-3.141592653589793, -3.141592653589793]

        """
        _, xvars = self._split(super().getPositions())
        return [v.evaluate(x, self._Lx) for v, x in zip(self._variables, xvars)]


class ExtendedSpaceContext(openmm.Context):
    """
    An extension of OpenMM's Context_ class.

    Parameters
    ----------
        variables : list(DynamicalVariable)
            The dynamical variables to be added to the system's phase space.
        system : openmm.System
            The System_ which will be simulated
        integrator : openmm.Integrator
            The Integrator_ which will be used to simulate the System_.
        platform : openmm.Platform
            The Platform_ to use for calculations.
        properties : dict(str: str)
            A set of values for platform-specific properties. Keys are the property names.

    Attributes
    ----------
        variables : list(DynamicalVariable)
            The dynamical variables added to the system's phase space.
        driving_forces : list[openmm.CustomCVForce]
            A list of CustomCVForce_ objects responsible for evaluating the potential energy terms
            which couples the extra dynamical variables to their associated collective
            variables.

    """

    def __init__(self, variables, system, *args, **kwargs):
        def get_driving_force(variables):
            driving_force = openmm.CustomCVForce(_get_energy_function(variables))
            for name, value in _get_parameters(variables).items():
                driving_force.addGlobalParameter(name, value)
            for var in variables:
                driving_force.addCollectiveVariable(var.id, var.force)
                for colvar in var.colvars:
                    driving_force.addCollectiveVariable(colvar.id, colvar.force)
            return driving_force

        driving_force_variables = [[]]
        for var in variables:
            if len(driving_force_variables[-1]) + len(var.colvars) > 31:
                driving_force_variables.append([])
            driving_force_variables[-1].append(deepcopy(var))
        driving_forces = [get_driving_force(vars) for vars in driving_force_variables]
        box_length_x = system.getDefaultPeriodicBoxVectors()[0].x
        num_particles = system.getNumParticles()
        collective_variables = []
        for vars in driving_force_variables:
            for var in vars:
                system.addParticle(var._particle_mass(box_length_x))
                var.force.setParticleParameters(0, num_particles, [])
                num_particles += 1
                collective_variables += var.colvars
        for force in system.getForces() + collective_variables:
            self._add_fake_particles(force, len(variables))
        for driving_force in driving_forces:
            system.addForce(driving_force)
        _update_RMSD_forces(system)
        super().__init__(system, *args, **kwargs)
        self.setParameter("Lx", box_length_x)
        self.variables = variables
        self.driving_forces = driving_forces

    def _add_fake_particles(self, force, n):
        if isinstance(force, openmm.NonbondedForce):
            for i in range(n):
                force.addParticle(0.0, 1.0, 0.0)
        elif isinstance(force, openmm.CustomNonbondedForce):
            for i in range(n):
                force.addParticle([0.0] * force.getNumPerParticleParameters())
        elif isinstance(force, openmm.CustomGBForce):
            parameter_list = list(map(force.getParticleParameters, range(force.getNumParticles())))
            force.addPerParticleParameter("isreal")
            for index in range(force.getNumEnergyTerms()):
                expression, type = force.getEnergyTermParameters(index)
                energy = "isreal*E_GB" if type == force.SingleParticle else "isreal1*isreal2*E_GB"
                force.setEnergyTermParameters(index, f"{energy}; E_GB={expression}", type)
            for index, parameters in enumerate(parameter_list):
                force.setParticleParameters(index, parameters + (1.0,))
            for i in range(n):
                force.addParticle(parameter_list[0] + (0.0,))
        elif isinstance(force, openmm.GBSAOBCForce):
            raise RuntimeError("GBSAOBCForce not supported")

    def getState(self, **kwargs):
        """
        Returns a :class:`ExtendedSpaceState` object.

        .. _getState: http://docs.openmm.org/latest/api-python/generated/openmm.openmm.Context.html#openmm.openmm.Context.getState  # noqa: E501

        Keyword Args
        ------------
            **kwargs
                See getState_.

        """
        return ExtendedSpaceState(self.variables, super().getState(**kwargs))

    def setPeriodicBoxVectors(self, a, b, c):
        """
        Set the vectors defining the axes of the periodic box.

        .. warning::
            Only orthorhombic boxes are allowed.

        Parameters
        ----------
            a : openmm.Vec3
                The vector defining the first edge of the periodic box.
            b : openmm.Vec3
                The vector defining the second edge of the periodic box.
            c : openmm.Vec3
                The vector defining the third edge of the periodic box.

        """
        a = openmm.Vec3(*map(_standardized, a))
        b = openmm.Vec3(*map(_standardized, b))
        c = openmm.Vec3(*map(_standardized, c))
        if not (a.y == a.z == b.x == b.z == c.x == c.y == 0.0):
            raise ValueError("Only orthorhombic boxes are allowed")
        self.setParameter("Lx", a.x)
        super().setPeriodicBoxVectors(a, b, c)
        system = self.getSystem()
        ntotal = system.getNumParticles()
        nvars = len(self.variables)
        for i, v in enumerate(self.variables):
            system.setParticleMass(ntotal - nvars + i, v._particle_mass(a.x))
        self.reinitialize(preserveState=True)

    def setPositions(self, positions, extended_positions=None):
        """
        Sets the positions of all particles and extended-space variables in
        this context. If the latter are not provided, then suitable values
        are automatically determined from the particle positions.

        Parameters
        ----------
            positions : list of openmm.Vec3
                The positions of physical particles.

        Keyword Args
        ------------
            extended_positions : list of float or unit.Quantity
                The positions of extended-space particles.

        """
        a, b, _ = self.getState().getPeriodicBoxVectors()
        nvars = len(self.variables)
        particle_positions = _standardized(positions)
        if extended_positions is None:
            extra_positions = [openmm.Vec3(0, b.y * (i + 1) / (nvars + 2), 0) for i in range(nvars)]
            minisystem = openmm.System()
            expression = _get_energy_function(self.variables)
            for i, v in enumerate(self.variables):
                expression += f"; {v.id}={v._get_energy_function(index=i+1)}"
            force = openmm.CustomCompoundBondForce(nvars, expression)
            force.addBond(range(nvars), [])
            for name, value in _get_parameters(self.variables).items():
                force.addGlobalParameter(name, value)
            force.addGlobalParameter("Lx", a.x)
            for v in self.variables:
                minisystem.addParticle(v._particle_mass(a.x))
                for cv in v.colvars:
                    value = cv.evaluate(self.getSystem(), particle_positions + extra_positions)
                    force.addGlobalParameter(cv.id, value)
            minisystem.addForce(force)
            minicontext = openmm.Context(
                minisystem,
                openmm.CustomIntegrator(0),
                openmm.Platform.getPlatformByName("Reference"),
            )
            minicontext.setPositions(extra_positions)
            openmm.LocalEnergyMinimizer.minimize(minicontext, 1 * unit.kilojoules_per_mole, 0)
            ministate = minicontext.getState(getPositions=True)
            extra_positions = ministate.getPositions().value_in_unit(unit.nanometers)
        else:
            extra_positions = [
                openmm.Vec3(x, b.y * (i + 1) / (nvars + 2), 0)
                for i, x in enumerate(extended_positions)
            ]
        super().setPositions(particle_positions + extra_positions)

    def setVelocitiesToTemperature(self, temperature, randomSeed=None):
        """
        Sets the velocities of all particles in the system to random values chosen from a
        Maxwell-Boltzmann distribution at a given temperature.

        .. warning ::
            The velocities of the extended-space variables are set to values consistent with the
            distribution at its own specified temperature.

        Parameters
        ----------
            temperature : float or unit.Quantity
                The temperature of the system.

        Keyword Args
        ------------
            randomSeed : int, default=None
                A seed for the random number generator.

        """
        system = self.getSystem()
        Ntotal = system.getNumParticles()
        Natoms = Ntotal - len(self.variables)
        m = np.array([system.getParticleMass(i) / unit.dalton for i in range(Ntotal)])
        T = np.array(
            [_standardized(temperature)] * Natoms + [v.temperature for v in self.variables]
        )
        sigma = np.sqrt(_standardized(unit.MOLAR_GAS_CONSTANT_R) * T / m)
        random_state = np.random.RandomState(randomSeed)
        velocities = sigma[:, np.newaxis] * random_state.normal(0, 1, (Ntotal, 3))
        super().setVelocities(velocities)


class ExtendedSpaceSimulation(app.Simulation):
    """
    A simulation involving extended phase-space variables.

    Parameters
    ----------
        variables : list of DynamicalVariable
            The dynamical variables to be added to the system's phase space.
        topology : openmm.app.Topology
            A Topology describing the system to be simulated.
        system : openmm.System
            The OpenMM System_ object to be simulated.
        integrator : openmm.Integrator
            The OpenMM Integrator to use for simulating the system dynamics.

    Keyword Args
    ------------
        platform : openmm.Platform, default=None
            If not None, the OpenMM Platform_ to use.
        platformProperties : dict, default=None
            If not None, a set of platform-specific properties to pass to the Context's constructor.

    """

    def __init__(
        self, variables, topology, system, integrator, platform=None, platformProperties=None
    ):
        self.variables = variables
        self._periodic_tasks = []

        for force in system.getForces():
            cls = force.__class__.__name__
            if cls == "CMMotionRemover" or cls.endswith("Barostat"):
                raise Exception("UFED: system cannot contain CMMotionRemover nor any Barostat")

        box_vectors = topology.getPeriodicBoxVectors()
        if box_vectors is None:
            raise Exception("UFED: system must be confined in a simulation box")

        self.topology = topology
        self.system = system
        self.integrator = integrator
        if openmm.__version__ < "7.7":
            self.currentStep = 0
        self.reporters = []
        self._usesPBC = True
        if platform is None:
            self.context = ExtendedSpaceContext(variables, system, integrator)
        elif platformProperties is None:
            self.context = ExtendedSpaceContext(variables, system, integrator, platform)
        else:
            self.context = ExtendedSpaceContext(
                variables, system, integrator, platform, platformProperties
            )

    def add_periodic_task(self, task, force_group=0):
        """
        Adds a task to be executed periodically along this simulation.

        Parameters
        ----------
            task : PeriodicTask
                A :class:`~ufedmm.ufedmm.PeriodicTask` object.

        """
        task.initialize(self)
        self._periodic_tasks.append(task)

    def step(self, steps):
        """
        Executed a given number of simulation steps.

        Parameters
        ----------
            steps : int
                The number of steps to be executed.

        """
        if isinstance(self.integrator, ufedmm.AbstractMiddleRespaIntegrator):
            if self.integrator._num_rattles == 0 and self.system.getNumConstraints() > 0:
                raise RuntimeError("Integrator cannot handle constraints")
        if self._periodic_tasks:
            for task in self._periodic_tasks:
                task.update(self, steps)
            self.reporters = self._periodic_tasks + self.reporters
            self._simulate(endStep=self.currentStep + steps)
            self.reporters = self.reporters[len(self._periodic_tasks) :]
        else:
            self._simulate(endStep=self.currentStep + steps)

    def saveCheckpoint(self, file):
        if isinstance(file, str):
            with open(file, "wb") as f:
                self.saveCheckpoint(f)
        else:
            for task in self._periodic_tasks:
                task.saveCheckpoint(file)
            file.write(self.context.createCheckpoint())

    def loadCheckpoint(self, file):
        if isinstance(file, str):
            with open(file, "rb") as f:
                self.loadCheckpoint(f)
        else:
            for task in self._periodic_tasks:
                task.loadCheckpoint(file, self.context)
            self.context.loadCheckpoint(file.read())


class UnifiedFreeEnergyDynamics(object):
    """
    A Unified Free-Energy Dynamics (UFED) setup.

    Parameters
    ----------
        variables : list of DynamicalVariable
            The variables.
        temperature : float or unit.Quantity
            The temperature.

    Keyword Args
    ------------
        height : float or unit.Quantity, default=None
            The height.
        frequency : int, default=None
            The frequency.
        bias_factor : float, default=None
            Scales the height of the hills added to the metadynamics bias potential. If it is
            `None`, then the hills will have a constant height over time. For a bias factor to be
            applicable, all bias variables must be at the same temperature T. The extended-space
            dynamical variables are sampled as if the effective temperature were T*bias_factor.
        grid_expansion : int, default=20
            The grid expansion.
        enforce_gridless : bool, default=False
            If this is `True`, gridless metadynamics is enforced even for 1D to 3D problems.
        buffer_size : int, default=100
            The buffer size.

    Example
    -------
        >>> import ufedmm
        >>> from openmm import unit
        >>> model = ufedmm.AlanineDipeptideModel(water='tip3p')
        >>> mass = 50*unit.dalton*(unit.nanometer/unit.radians)**2
        >>> Ks = 1000*unit.kilojoules_per_mole/unit.radians**2
        >>> Ts = 1500*unit.kelvin
        >>> limit = 180*unit.degrees
        >>> s_phi = ufedmm.DynamicalVariable('s_phi', -limit, limit, mass, Ts, model.phi, Ks)
        >>> s_psi = ufedmm.DynamicalVariable('s_psi', -limit, limit, mass, Ts, model.psi, Ks)
        >>> ufedmm.UnifiedFreeEnergyDynamics([s_phi, s_psi], 300*unit.kelvin)
        <variables=[s_phi, s_psi], temperature=300, height=None, frequency=None>

    """

    def __init__(
        self,
        variables,
        temperature,
        height=None,
        frequency=None,
        bias_factor=None,
        grid_expansion=20,
        enforce_gridless=False,
        buffer_size=100,
    ):
        self.variables = variables
        self.temperature = _standardized(temperature)
        self.height = _standardized(height)
        self.frequency = frequency
        self.bias_factor = bias_factor
        self.grid_expansion = grid_expansion
        self.enforce_gridless = enforce_gridless
        self.buffer_size = buffer_size

        dimensions = sum(v.sigma is not None for v in variables)
        self._metadynamics = not (dimensions == 0 or height is None or frequency is None)

    def __repr__(self):
        properties = (
            f"temperature={self.temperature}, height={self.height}, frequency={self.frequency}"
        )
        return f'<variables=[{", ".join(v.id for v in self.variables)}], {properties}>'

    def __getstate__(self):
        return dict(
            variables=self.variables,
            temperature=self.temperature,
            height=self.height,
            frequency=self.frequency,
            bias_factor=self.bias_factor,
            grid_expansion=self.grid_expansion,
            enforce_gridless=self.enforce_gridless,
            buffer_size=self.buffer_size,
        )

    def __setstate__(self, kw):
        self.__init__(**kw)

    def simulation(self, topology, system, integrator, platform=None, platformProperties=None):
        """
        Returns a ExtendedSpaceSimulation object.

        .. warning::
            If the temperature of any driving parameter is different from the particle-system
            temperature, then the passed integrator must be a CustomIntegrator_ object
            containing a per-dof variable `kT` whose content is the Boltzmann constant times
            the temperature associated to each degree of freedom.
            This is true for all integrators available in :mod:`ufedmm.integrators`, which are
            subclasses of :class:`ufedmm.integrators.CustomIntegrator`.

        Parameters
        ----------
            topology : openmm.app.Topology
                The topology.
            system : openmm.System
                The system.
            integrator :
                The integrator.

        Keyword Args
        ------------
            platform : openmm.Platform, default=None
                The platform.
            platformProperties : dict, default=None
                The platform properties.

        Example
        -------
            >>> import ufedmm
            >>> from openmm import unit
            >>> model = ufedmm.AlanineDipeptideModel(water='tip3p')
            >>> mass = 50*unit.dalton*(unit.nanometer/unit.radians)**2
            >>> Ks = 1000*unit.kilojoules_per_mole/unit.radians**2
            >>> Ts = 1500*unit.kelvin
            >>> dt = 2*unit.femtoseconds
            >>> gamma = 10/unit.picoseconds
            >>> limit = 180*unit.degrees
            >>> s_phi = ufedmm.DynamicalVariable('s_phi', -limit, limit, mass, Ts, model.phi, Ks)
            >>> s_psi = ufedmm.DynamicalVariable('s_psi', -limit, limit, mass, Ts, model.psi, Ks)
            >>> ufed = ufedmm.UnifiedFreeEnergyDynamics([s_phi, s_psi], 300*unit.kelvin)
            >>> integrator = ufedmm.GeodesicLangevinIntegrator(300*unit.kelvin, gamma, dt)
            >>> simulation = ufed.simulation(model.topology, model.system, integrator)

        """
        simulation = ExtendedSpaceSimulation(
            self.variables,
            topology,
            system,
            integrator,
            platform,
            platformProperties,
        )

        if self._metadynamics:
            simulation.add_periodic_task(
                _Metadynamics(
                    self.variables,
                    self.height,
                    self.frequency,
                    bias_factor=self.bias_factor,
                    buffer_size=self.buffer_size,
                    grid_expansion=self.grid_expansion,
                    enforce_gridless=self.enforce_gridless,
                ),
            )

        if any(v.temperature != self.temperature for v in self.variables):
            ntotal = system.getNumParticles()
            nvars = len(self.variables)
            simulation.context.setPositions([openmm.Vec3(0, 0, 0)] * (ntotal - nvars), [0] * nvars)
            vartemps = [v.temperature for v in self.variables]
            if isinstance(integrator, ufedmm.integrators.CustomIntegrator):
                integrator.update_temperatures(self.temperature, vartemps)
            else:
                temperatures = [self.temperature] * (ntotal - len(vartemps)) + vartemps
                kT = [unit.MOLAR_GAS_CONSTANT_R * T * openmm.Vec3(1, 1, 1) for T in temperatures]
                integrator.setPerDofVariableByName("kT", kT)

        return simulation
