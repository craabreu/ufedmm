"""
.. module:: ufedmm
   :platform: Unix, Windows
   :synopsis: Unified Free Energy Dynamics with OpenMM

.. moduleauthor:: Charlles Abreu <abreu@eq.ufrj.br>

.. _Context: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.Context.html
.. _CustomCVForce: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.CustomCVForce.html
.. _CustomIntegrator: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.CustomIntegrator.html
.. _Force: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.Force.html
.. _Platform: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.Platform.html
.. _System: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.System.html

"""

import functools
import io
from copy import deepcopy

import numpy as np
from simtk import openmm, unit
from simtk.openmm import app


def _standardize(quantity):
    if unit.is_quantity(quantity):
        return quantity.value_in_unit_system(unit.md_unit_system)
    else:
        return quantity


class CollectiveVariable(object):
    """
    A function of particle coordinates evaluated by means of an OpenMM Force_ object.

    Parameters
    ----------
        id : str
            A valid identifier string for this collective variable.
        force : openmm.Force
            An OpenMM Force_ object whose energy function is used to evaluate this collective
            variable.

    Example
    -------
        >>> import ufedmm
        >>> from simtk import openmm, unit
        >>> cv = ufedmm.CollectiveVariable('psi', openmm.CustomTorsionForce('theta'))
        >>> cv.force.addTorsion(0, 1, 2, 3, [])
        0

    """
    def __init__(self, id, force):
        if not id.isidentifier():
            raise ValueError('Parameter id must be a valid variable identifier')
        self.id = id
        self.force = force

    def _create_context(self, system, positions):
        system_copy = deepcopy(system)
        for force in system_copy.getForces():
            force.setForceGroup(0)
        force_copy = deepcopy(self.force)
        force_copy.setForceGroup(1)
        system_copy.addForce(force_copy)
        platform = openmm.Platform.getPlatformByName('Reference')
        context = openmm.Context(system_copy, openmm.CustomIntegrator(0), platform)
        context.setPositions(positions)
        return context

    def evaluate(self, system, positions):
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

        Returns
        -------
            float

        Example
        -------
            >>> import ufedmm
            >>> from simtk import unit
            >>> model = ufedmm.AlanineDipeptideModel()
            >>> model.phi.evaluate(model.system, model.positions)
            3.141592653589793
            >>> model.psi.evaluate(model.system, model.positions)
            3.141592653589793

        """
        context = self._create_context(system, positions)
        energy = context.getState(getEnergy=True, groups={1}).getPotentialEnergy()
        return energy.value_in_unit(unit.kilojoules_per_mole)

    def effective_mass(self, system, positions):
        """
        Computes the effective mass of the collective variable for a given system and a given set of
        particle coordinates.

        Parameters
        ----------
            system : openmm.System
                The system for which the collective variable will be evaluated.
            positions : list of openmm.Vec3
                A list whose size equals the number of particles in the system and which contains
                the coordinates of these particles.

        Returns
        -------
            float

        Example
        -------
            >>> import ufedmm
            >>> from simtk import unit
            >>> model = ufedmm.AlanineDipeptideModel()
            >>> model.phi.effective_mass(model.system, model.positions)
            0.0479588726559707
            >>> model.psi.effective_mass(model.system, model.positions)
            0.05115582071188152

        """
        context = self._create_context(system, positions)
        forces = _standardize(context.getState(getForces=True, groups={1}).getForces(asNumpy=True))
        summation = 0.0
        for i, f in enumerate(forces):
            m = system.getParticleMass(i).value_in_unit(unit.dalton)
            summation += f.dot(f)/m
        return 1.0/summation


class DynamicalVariable(object):
    """
    An extended-space variable whose dynamics is coupled to that of one of more collective variables
    of a system.

    Parameters
    ----------
        id : str
            A valid identifier string.
        min_value : float or unit.Quantity
            The minimum allowable value.
        max_value : float or unit.Quantity
            The maximum allowable value.
        mass : float or unit.Quantity
            The mass of this dynamical variable.
        temperature : float or unit.Quantity
            The temperature of this .
        colvar : :class:`~ufedmm.ufedmm.CollectiveVariable`
            A colective variable.
        potential : float or unit.Quantity or str
            Either the value of the force constant of a harmonic driving force or an algebraic
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
        >>> import ufedmm
        >>> from simtk import openmm, unit
        >>> cv = ufedmm.CollectiveVariable('psi', openmm.CustomTorsionForce('theta'))
        >>> cv.force.addTorsion(0, 1, 2, 3, [])
        0
        >>> mass = 50*unit.dalton*(unit.nanometer/unit.radians)**2
        >>> K = 1000*unit.kilojoules_per_mole/unit.radians**2
        >>> Ts = 1500*unit.kelvin
        >>> ufedmm.DynamicalVariable('s_psi', -180*unit.degrees, 180*unit.degrees, mass, Ts, cv, K)
        <s_psi in [-3.141592653589793, 3.141592653589793], periodic, m=50, T=1500>

    """
    def __init__(self, id, min_value, max_value, mass, temperature, colvar, potential,
                 periodic=True, sigma=None, grid_size=None, **parameters):
        self.id = id
        self.min_value = _standardize(min_value)
        self.max_value = _standardize(max_value)
        self._range = self.max_value - self.min_value
        self.mass = _standardize(mass)
        self.temperature = _standardize(temperature)
        self.colvar = colvar

        if isinstance(potential, str):
            self.potential = potential
            self.parameters = {key: _standardize(value) for key, value in parameters.items()}
        else:
            id = self.colvar.id
            if periodic:
                self.potential = f'0.5*K_{id}*min(d{id},{self._range}-d{id})^2'
                self.potential += f'; d{id}=abs({id}-{self.id})'
            else:
                self.potential = f'0.5*K_{id}*({id}-{self.id})^2'
            self.parameters = {f'K_{id}': _standardize(potential)}

        self.periodic = periodic

        if sigma is None:
            self.sigma = self.grid_size = None
        else:
            self.sigma = _standardize(sigma)
            self._scaled_variance = (self.sigma/self._range)**2
            if grid_size is None:
                self.grid_size = int(np.ceil(5*self._range/self.sigma)) + 1
            else:
                self.grid_size = grid_size

        self.force = openmm.CustomExternalForce(self.get_energy_function())
        self.force.addGlobalParameter('Lx', 0.0)
        self.force.addParticle(0, [])

    def __repr__(self):
        properties = f'm={self.mass}, T={self.temperature}'
        status = 'periodic' if self.periodic else 'non-periodic'
        return f'<{self.id} in [{self.min_value}, {self.max_value}], {status}, {properties}>'

    def __getstate__(self):
        return dict(
            id=self.id,
            min_value=self.min_value,
            max_value=self.max_value,
            mass=self.mass,
            temperature=self.temperature,
            colvar=self.colvar,
            potential=self.potential,
            periodic=self.periodic,
            sigma=self.sigma,
            grid_size=self.grid_size,
            **self.parameters,
        )

    def __setstate__(self, kw):
        self.__init__(**kw)

    def _particle_mass(self, Lx):
        length = Lx if self.periodic else Lx/2
        return self.mass*(self._range/length)**2

    def _particle_position(self, value, Lx, y=0):
        length = Lx if self.periodic else Lx/2
        return openmm.Vec3(length*(value - self.min_value)/self._range, y, 0)*unit.nanometer

    def get_energy_function(self, index=''):
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
            energy = f'{self.min_value}+{self._range}*(x{index}/Lx-floor(x{index}/Lx))'
        else:
            energy = f'{self.min_value}+{2*self._range}*min(pos{index},1-pos{index})'
            energy += f';pos{index}=x{index}/Lx-floor(x{index}/Lx)'
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
        pos = x/Lx - np.floor(x/Lx)
        if self.periodic:
            return self.min_value + self._range*pos
        else:
            return self.min_value + 2*self._range*min(pos, 1-pos)


class DynamicalVariableTuple(tuple):
    def get_energy_function(self):
        energies = [v.potential.split(';', 1) for v in self]
        energy_terms = [energy[0] for energy in energies]
        definitions = [energy[1] for energy in energies if len(energy) == 2]
        expression = ';'.join(['+'.join(energy_terms)] + list(definitions))
        return expression

    def get_parameters(self):
        parameters = {}
        for v in self:
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
        steps = self.frequency - simulation.currentStep % self.frequency
        return (steps, True, False, False, False, False)

    def report(self, simulation, state):
        pass


class Metadynamics(PeriodicTask):
    """
    Extended-space Metadynamics.

    Parameters
    ----------
        variables : list of DynamicalVariable
            The variables.
        temperature : float or unit.Quantity
            The temperature.
        height : float or unit.Quantity, default=None
            The height.
        frequency : int, default=None
            The frequency.

    Keyword Args
    ------------
        buffer_size : int, default=100
            The buffer size.
        grid_expansion : int, default=20
            The number of extra grid points to be used in periodic directions of multidimensional
            tabulated functions. This aims at avoiding boundary discontinuity artifacts.
        enforce_gridless : bool, default=False
            Enforce gridless metadynamics even for 1D to 3D problems.

    """
    def __init__(self, variables, height, frequency, buffer_size=100, grid_expansion=20,
                 enforce_gridless=False):
        super().__init__(frequency)
        self.bias_indices = [i for i, v in enumerate(variables) if v.sigma is not None]
        self.bias_variables = DynamicalVariableTuple(variables[i] for i in self.bias_indices)
        self.height = _standardize(height)
        self.buffer_size = buffer_size
        self.grid_expansion = grid_expansion
        self._use_grid = len(self.bias_variables) < 4 and not enforce_gridless
        if self._use_grid:
            self.force = self._interpolation_grid_force()
        else:
            self.force = self._hills_force()

    def _interpolation_grid_force(self):
        self._widths = []
        self._bounds = []
        self._extra_points = []
        for v in self.bias_variables:
            extra_points = min(self.grid_expansion, v.grid_size) if v.periodic else 0
            extra_range = extra_points*v._range/(v.grid_size - 1)
            self._widths.append(v.grid_size + 2*extra_points)
            self._bounds += [v.min_value - extra_range, v.max_value + extra_range]
            self._extra_points.append(extra_points)
        self._bias = np.zeros(np.prod(self._widths))
        num_bias_variables = len(self.bias_variables)
        if num_bias_variables == 1:
            self._table = openmm.Continuous1DFunction(self._bias, *self._bounds)
        elif num_bias_variables == 2:
            self._table = openmm.Continuous2DFunction(*self._widths, self._bias, *self._bounds)
        else:
            self._table = openmm.Continuous3DFunction(*self._widths, self._bias, *self._bounds)
        expression = f'bias({",".join(v.id for v in self.bias_variables)})'
        for i, v in enumerate(self.bias_variables):
            expression += f';{v.id}={v.get_energy_function(i+1)}'
        force = openmm.CustomCVForce(expression)
        for i in range(num_bias_variables):
            x = openmm.CustomExternalForce('x')
            x.addParticle(0, [])
            force.addCollectiveVariable(f'x{i+1}', x)
        force.addTabulatedFunction('bias', self._table)
        force.addGlobalParameter('Lx', 0)
        return force

    def _hills_force(self):
        num_bias_variables = len(self.bias_variables)
        centers = [f'center{i+1}' for i in range(num_bias_variables)]
        exponents = []
        for v, center in zip(self.bias_variables, centers):
            if v.periodic:  # von Mises
                factor = 2*np.pi/v._range
                exponents.append(f'{1.0/(factor*v.sigma)**2}*(cos({factor}*({v.id}-{center}))-1)')
            else:  # Gauss
                exponents.append(f'({-0.5/v.sigma**2})*({v.id}-{center})^2')
        expression = f'height*exp({"+".join(exponents)})'
        for i, v in enumerate(self.bias_variables):
            expression += f';{v.id}={v.get_energy_function(i+1)}'
        force = openmm.CustomCompoundBondForce(num_bias_variables, expression)
        force.addPerBondParameter('height')
        for center in centers:
            force.addPerBondParameter(center)
        force.addGlobalParameter('Lx', 0)
        return force

    def _add_buffer(self, simulation):
        size = min(self.buffer_size, self._total_hills - self.force.getNumBonds())
        for i in range(size):
            self.force.addBond(self.particles, [0]*(len(self.bias_variables)+1))
        simulation.context.reinitialize(preserveState=True)

    def add_bias(self, simulation, bias):
        if self._use_grid:
            self._bias += bias.ravel()
            if len(self.bias_variables) == 1:
                self._table.setFunctionParameters(self._bias, *self._bounds)
            else:
                self._table.setFunctionParameters(*self._widths, self._bias, *self._bounds)
            self.force.updateParametersInContext(simulation.context)

    def initialize(self, simulation, force_group):
        self.particles = [simulation.get_num_particles() + index for index in self.bias_indices]
        self.Lx = simulation.context.getParameter('Lx')
        if self._use_grid:
            for i, particle in enumerate(self.particles):
                self.force.getCollectiveVariable(i).setParticleParameters(0, particle, [])
        else:
            self._num_hills = 0
        self.force.setForceGroup(force_group)
        simulation.system.addForce(self.force)
        simulation.context.reinitialize(preserveState=True)

    def update(self, simulation, steps):
        if not self._use_grid:
            steps_until_next_report = self.frequency - simulation.currentStep % self.frequency
            if steps_until_next_report > steps:
                required_hills = 0
            else:
                required_hills = (steps - steps_until_next_report)//self.frequency + 1
            self._total_hills = self._num_hills + required_hills
            self._add_buffer(simulation)

    def report(self, simulation, state):
        coords = state.getPositions()
        xcoords = [coords[index].x for index in self.particles]
        centers = [v.evaluate(x, self.Lx) for v, x in zip(self.bias_variables, xcoords)]
        if self._use_grid:
            hills = []
            for i, v in enumerate(self.bias_variables):
                x = (centers[i] - v.min_value)/v._range
                dist = np.linspace(0, 1, num=v.grid_size) - x
                if v.periodic:  # von Mises
                    exponents = (np.cos(2*np.pi*dist)-1)/(4*np.pi*np.pi*v._scaled_variance)
                else:  # Gauss
                    exponents = -0.5*dist*dist/v._scaled_variance
                values = self.height*np.exp(exponents)
                if v.periodic:
                    n = self._extra_points[i] + 1
                    values = np.hstack((values[-n:-1], values, values[1:n]))
                hills.append(values)
            if len(self.bias_variables) == 1:
                bias = hills[0]
            else:
                bias = functools.reduce(np.multiply.outer, reversed(hills))
            self.add_bias(simulation, bias)
        else:
            if self._num_hills == self.force.getNumBonds():
                self._add_buffer(simulation)
            self.force.setBondParameters(self._num_hills, self.particles, [self.height] + centers)
            self._num_hills += 1
            self.force.updateParametersInContext(simulation.context)


class ExtendedSpaceSimulation(app.Simulation):
    """
    A simulation involving extended phase-space variables.

    Parameters
    ----------
        variables : list of DynamicalVariable
            The dynamical variables to be added to the system's phase space.
        topology : openmm.app.Topology
            A Topology describing the the system to simulate.
        system : openmm.System
            The OpenMM System_ object to simulate.
        integrator : openmm.Integrator
            The OpenMM Integrator to use for simulating the System.

    Keyword Args
    ------------
        platform : openmm.Platform, default=None
            If not None, the OpenMM Platform_ to use.
        platformProperties : dict, default=None
            If not None, a set of platform-specific properties to pass to the Context's constructor

    """
    def __init__(self, variables, topology, system, integrator, platform=None, platformProperties=None):
        self.variables = DynamicalVariableTuple(variables)
        self._periodic_tasks = []

        for force in system.getForces():
            cls = force.__class__.__name__
            if cls == 'CMMotionRemover' or cls.endswith('Barostat'):
                raise Exception('UFED: system cannot contain CMMotionRemover nor any Barostat')

        Vx, Vy, Vz = topology.getPeriodicBoxVectors()
        if not (Vx.y == Vx.z == Vy.x == Vy.z == Vz.x == Vz.y == 0.0):
            raise ValueError('UFED: only orthorhombic boxes are allowed')

        self.driving_force = openmm.CustomCVForce(self.variables.get_energy_function())
        for name, value in self.variables.get_parameters().items():
            self.driving_force.addGlobalParameter(name, value)
        for v in self.variables:
            self.driving_force.addCollectiveVariable(v.id, deepcopy(v.force))
            self.driving_force.addCollectiveVariable(v.colvar.id, deepcopy(v.colvar.force))

        positions = [openmm.Vec3(0, 0, 0) for atom in topology.atoms()]
        modeller = app.Modeller(topology, positions)
        extra_atom = 'ATOM      1  Cs   Cs A   1       0.000   0.000   0.000  1.00  0.00'
        pdb = app.PDBFile(io.StringIO(extra_atom))
        for i in range(len(self.variables)):
            modeller.add(pdb.topology, pdb.positions)
        self._num_particles = system.getNumParticles()
        nb_types = (openmm.NonbondedForce, openmm.CustomNonbondedForce)
        nb_forces = [f for f in system.getForces() if isinstance(f, nb_types)]
        for i, v in enumerate(self.variables):
            system.addParticle(v._particle_mass(Vx.x))
            for nb_force in nb_forces:
                if isinstance(nb_force, openmm.NonbondedForce):
                    nb_force.addParticle(0.0, 1.0, 0.0)
                else:
                    nb_force.addParticle([0.0]*nb_force.getNumPerParticleParameters())
            parameter = self.driving_force.getCollectiveVariable(2*i)
            parameter.setParticleParameters(0, self._num_particles+i, [])
        system.addForce(self.driving_force)

        super().__init__(modeller.topology, system, integrator, platform, platformProperties)
        self.context.setParameter('Lx', Vx.x)

    def add_periodic_task(self, task, force_group=0):
        """
        Adds a task to be executed periodically along this simulation.

        Parameters
        ----------
            task : PeriodicTask
                A :class:`~ufedmm.ufedmm.PeriodicTask` object.

        Keyword Args
        ------------
            force_group : int, default=0
                The force group to add new forces to.

        """
        task.initialize(self, force_group)
        self._periodic_tasks.append(task)

    def get_num_particles(self):
        """
        Returns the number of real particles in a simulation.

        Returns
        -------
            int

        """
        return self._num_particles

    def set_positions(self, positions, extended=False, **kwargs):
        """
        Sets the positions of all particles in the simulation's context.

        Parameters
        ----------
            positions : list of openmm.Vec3
                The positions of all particles, which may include the extra particles that represent
                the extended-space variables (see below).

        Keyword Args
        ------------
            extended : bool, default=False
                Whether `positions` include those of the extra particles that represent the
                extended-space variables.
            **kwargs
                Identifiers and values to be assigned to the dynamical variables. For those which
                are not specified, the value will be made equal to that of the associated collective
                variables. Note that these keyword arguments will have no effect whatsoever if
                `extended=False`.

        """
        if extended:
            self.context.setPositions(positions)
        else:
            extended_positions = deepcopy(positions)
            n = len(extended_positions)
            for i in range(len(self.variables)):
                extended_positions.append(openmm.Vec3(0, 0, 0)*unit.nanometer)
            Vx, Vy, _ = self.context.getState().getPeriodicBoxVectors()
            for i, v in enumerate(self.variables):
                y = Vy.y*(i + 1)/(len(self.variables) + 2)
                value = kwargs.get(v.id, v.colvar.evaluate(self.system, extended_positions))
                extended_positions[n+i] = v._particle_position(value, Vx.x, y)
            self.context.setPositions(extended_positions)

    def set_velocities_to_temperature(self, temperature, random_seed=None):
        """
        Sets the velocities of all particles in the system to random values chosen from a Boltzmann
        distribution at a given temperature.

        .. warning ::
            The velocities of the extended-space variables are set to zero.

        Parameters
        ----------
            temperature : float or unit.Quantity
                The temperature of the system.

        Keyword Args
        ------------
            random_seed : int, default=None
                A seed for the random number generator.

        """
        n = self.system.getNumParticles() - len(self.variables)
        masses = []
        for i, v in enumerate(self.variables):
            masses.append(self.system.getParticleMass(n+i))
            self.system.setParticleMass(n+i, 0)
        if random_seed is None:
            self.context.setVelocitiesToTemperature(temperature)
        else:
            self.context.setVelocitiesToTemperature(temperature, random_seed)
        for i, mass in enumerate(masses):
            self.system.setParticleMass(n+i, mass)

    def step(self, steps):
        """
        Executed a given number of simulation steps.

        Parameters
        ----------
            steps : int
                The number of steps to be executed.

        """
        if self._periodic_tasks:
            for task in self._periodic_tasks:
                task.update(self, steps)
                self.reporters.append(task)
            self._simulate(endStep=self.currentStep+steps)
            self.reporters = self.reporters[:-len(self._periodic_tasks)]
        else:
            self._simulate(endStep=self.currentStep+steps)


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
        grid_expansion : int, default=20
            The grid expansion.
        enforce_gridless : bool, default=False
            For gridless metadynamics even for 1D to 3D problems.
        buffer_size : int, default=100
            The buffer size.

    Properties:
        metadynamics : PeriodicTask
            If not `None`, it is the periodic task used to add hills to the bias potential.

    Example
    -------
        >>> import ufedmm
        >>> from simtk import unit
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
    def __init__(self, variables, temperature, height=None, frequency=None,
                 grid_expansion=20, enforce_gridless=False, buffer_size=100):
        self.variables = DynamicalVariableTuple(variables)
        self.temperature = _standardize(temperature)
        self.height = _standardize(height)
        self.frequency = frequency
        self.grid_expansion = grid_expansion
        self.enforce_gridless = enforce_gridless
        self.buffer_size = buffer_size

        dimension = sum(v.sigma is not None for v in variables)
        self.metadynamics = not (dimension == 0 or height is None or frequency is None)

    def __repr__(self):
        properties = f'temperature={self.temperature}, height={self.height}, frequency={self.frequency}'
        return f'<variables=[{", ".join(v.id for v in self.variables)}], {properties}>'

    def __getstate__(self):
        return dict(
            variables=self.variables,
            temperature=self.temperature,
            height=self.height,
            frequency=self.frequency,
            grid_expansion=self.grid_expansion,
            enforce_gridless=self.enforce_gridless,
            buffer_size=self.buffer_size,
        )

    def __setstate__(self, kw):
        self.__init__(**kw)

    def simulation(self, topology, system, integrator, platform=None, platformProperties=None):
        """
        Returns a Simulation.

        .. warning::
            If the temperature of any driving parameter is different from the particle-system
            temperature, then the passed integrator must be a CustomIntegrator object containing
            a per-dof variable called `kT`. The content of this variable is the Boltzmann constant
            times the temperature associated to each degree of freedom. One can employ, for
            instance, a :class:`~ufedmm.integrators.GeodesicBAOABIntegrator`.

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
            >>> from simtk import unit
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
            >>> integrator = ufedmm.GeodesicBAOABIntegrator(300*unit.kelvin, gamma, dt)
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

        if self.metadynamics:
            simulation.add_periodic_task(
                Metadynamics(
                    self.variables,
                    self.height,
                    self.frequency,
                    buffer_size=self.buffer_size,
                    grid_expansion=self.grid_expansion,
                    enforce_gridless=self.enforce_gridless,
                ),
            )

        if any(v.temperature != self.temperature for v in self.variables):
            simulation.context.setPositions([openmm.Vec3(0, 0, 0)]*system.getNumParticles())
            try:
                kT = integrator.getPerDofVariableByName('kT')
            except Exception:
                raise ValueError('CustomIntegrator with per-dof variable `kT` required')
            kB = _standardize(unit.MOLAR_GAS_CONSTANT_R)
            nparticles = system.getNumParticles() - len(self.variables)
            for i in range(nparticles):
                kT[i] = kB*self.temperature*openmm.Vec3(1, 1, 1)
            for i, v in enumerate(self.variables):
                kT[nparticles+i] = kB*v.temperature*openmm.Vec3(1, 0, 0)
            integrator.setPerDofVariableByName('kT', kT)

        return simulation
