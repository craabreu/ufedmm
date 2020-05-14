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
    A collective variable.

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

    def evaluate(self, system, positions):
        """
        Computes the value of the collective variable for a given system and set of particle
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
        new_system = openmm.System()
        new_system.setDefaultPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
        for index in range(len(positions)):
            new_system.addParticle(system.getParticleMass(index))
        new_system.addForce(deepcopy(self.force))
        platform = openmm.Platform.getPlatformByName('Reference')
        context = openmm.Context(new_system, openmm.CustomIntegrator(0), platform)
        context.setPositions(positions)
        energy = context.getState(getEnergy=True).getPotentialEnergy()
        return energy.value_in_unit(unit.kilojoules_per_mole)


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

        self.force = openmm.CustomExternalForce(self.expression())
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
        return openmm.Vec3(length*(value - self.min_value)/self._range, y, 0)

    def expression(self, index=''):
        if self.periodic:
            energy = f'{self.min_value}+{self._range}*(x{index}/Lx-floor(x{index}/Lx))'
        else:
            ramp_up = f'{self.min_value}+{2*self._range}*pos{index}'
            ramp_down = f'{self.max_value+self._range}-{2*self._range}*pos{index}'
            energy = f'select(step(0.5-pos),{ramp_up},{ramp_down})'
            energy += f'; pos{index}=x{index}/Lx-floor(x{index}/Lx)'
        return energy

    def evaluate(self, x, Lx):
        pos = x/Lx - np.floor(x/Lx)
        if self.periodic:
            return self.min_value + self._range*pos
        elif pos < 0.5:
            return self.min_value + 2*self._range*pos
        else:
            return self.max_value + self._range*(1 - 2*pos)


class DynamicalVariableTuple(tuple):
    def get_energy_function(self):
        energies = [v.potential.split(';', 1) for v in self]
        energy_terms, definitions = zip(*energies)
        expression = ';'.join(['+'.join(energy_terms)] + list(definitions))
        return expression

    def get_parameters(self):
        parameters = {}
        for v in self:
            parameters.update(v.parameters)
        return parameters


class GridlessMetadynamics(object):
    """
    Extended-space Metadynamics.

    """
    def __init__(self, variables, height, frequency, buffer_size=100):
        self.bias_indices = [i for i, v in enumerate(variables) if v.sigma is not None]
        self.bias_variables = DynamicalVariableTuple(variables[i] for i in self.bias_indices)
        self.height = height
        self.frequency = frequency
        self.buffer_size = buffer_size
        exponents = []
        definitions = []
        for i, v in enumerate(self.bias_variables):
            if v.periodic:  # von Mises
                factor = 2*np.pi/v._range
                exponents.append(f'{1.0/(factor*v.sigma)**2}*(cos({factor}*(v{i+1}-center{i+1}))-1)')
            else:  # Gauss
                exponents.append(f'{0.5/v.sigma**2}*(v{i+1}-center{i+1})^2')
            definitions.append(f'v{i+1}={v.expression(i+1)}')
        hill = ';'.join([f'height*exp({"+".join(exponents)})'] + definitions)
        n = len(self.bias_variables)
        self.force = openmm.CustomCompoundBondForce(n, hill)
        self.force.addGlobalParameter('Lx', 0)
        self.force.addPerBondParameter('height')
        for i in range(n):
            self.force.addPerBondParameter(f'center{i+1}')
        self._num_hills = 0

    def _add_buffer(self, simulation):
        size = min(self.buffer_size, self._total_hills - self.force.getNumBonds())
        for i in range(size):
            self.force.addBond(self.particles, [0]*(len(self.bias_variables)+1))
        simulation.context.reinitialize(preserveState=True)

    def initialize(self, simulation):
        num_particles = simulation.system.getNumParticles() - len(simulation.variables)
        self.particles = [num_particles + index for index in self.bias_indices]
        self.Lx = simulation.context.getParameter('Lx')
        simulation.system.addForce(self.force)

    def update(self, simulation, steps):
        steps_until_next_report = self.frequency - simulation.currentStep % self.frequency
        self._total_hills = self._num_hills + (steps - steps_until_next_report)//self.frequency
        self._add_buffer(simulation)

    def describeNextReport(self, simulation):
        steps = self.frequency - simulation.currentStep % self.frequency
        return (steps, True, False, False, False, False)

    def report(self, simulation, state):
        positions = state.getPositions()
        xcoords = [positions[i].x for i in self.particles]
        centers = [v.evaluate(x, self.Lx) for x, v in zip(xcoords, self.bias_variables)]
        if self._num_hills == self.force.getNumBonds():
            self._add_buffer(simulation)
        self.force.setBondParameters(self._num_hills, self.particles, [self.height] + centers)
        self._num_hills += 1
        self.force.updateParametersInContext(simulation.context)


class GriddedMetadynamics(object):
    """
    Extended-space Metadynamics.

    """
    def __init__(self, variables, height, frequency, grid_expansion):
        self.bias_indices = [i for i, v in enumerate(variables) if v.sigma is not None]
        self.bias_variables = [variables[i] for i in self.bias_indices]
        self.height = height
        self.frequency = frequency
        self.grid_expansion = grid_expansion
        self._widths = []
        self._bounds = []
        self._expanded = []
        self._extra_points = []
        for v in self.bias_variables:
            expanded = v.periodic  # and len(self.bias_variables) > 1
            extra_points = min(grid_expansion, v.grid_size) if expanded else 0
            extra_range = extra_points*v._range/(v.grid_size - 1)
            self._widths += [v.grid_size + 2*extra_points]
            self._bounds += [v.min_value - extra_range, v.max_value + extra_range]
            self._expanded += [expanded]
            self._extra_points += [extra_points]
        self._bias = np.zeros(np.prod(self._widths))
        if len(variables) == 1:
            self._table = openmm.Continuous1DFunction(
                self._bias, *self._bounds,  # self.bias_variables[0].periodic,
            )
        elif len(variables) == 2:
            self._table = openmm.Continuous2DFunction(*self._widths, self._bias, *self._bounds)
        elif len(variables) == 3:
            self._table = openmm.Continuous3DFunction(*self._widths, self._bias, *self._bounds)
        else:
            raise ValueError('UFED requires 1, 2, or 3 biased collective variables')
        parameter_list = ', '.join(f'{v.id}' for v in self.bias_variables)
        self.force = openmm.CustomCVForce(f'bias({parameter_list})')
        for v in self.bias_variables:
            self.force.addCollectiveVariable(v.id, deepcopy(v.force))
        self.force.addTabulatedFunction('bias', self._table)

    def add_bias(self, bias):
        self._bias += bias.flatten()
        if len(self.bias_variables) == 1:
            self._table.setFunctionParameters(self._bias, *self._bounds)
        else:
            self._table.setFunctionParameters(*self._widths, self._bias, *self._bounds)

    def initialize(self, simulation):
        num_particles = simulation.system.getNumParticles() - len(simulation.variables)
        for i, index in enumerate(self.bias_indices):
            parameter = self.force.getCollectiveVariable(i)
            parameter.setParticleParameters(0, num_particles+index, [])
        simulation.system.addForce(self.force)
        simulation.context.reinitialize(preserveState=True)

    def update(self, simulation, steps):
        pass

    def describeNextReport(self, simulation):
        steps = self.frequency - simulation.currentStep % self.frequency
        return (steps, False, False, False, False, False)

    def report(self, simulation, state):
        position = self.force.getCollectiveVariableValues(simulation.context)
        hills = []
        for i, v in enumerate(self.bias_variables):
            x = (position[i] - v.min_value)/v._range
            dist = np.abs(np.linspace(0, 1, num=v.grid_size) - x)
            if v.periodic:
                values = np.exp((np.cos(2*np.pi*dist)-1)/(4*np.pi**2*v._scaled_variance))  # von Mises
            else:
                values = np.exp(-0.5*dist*dist/v._scaled_variance)  # Gauss
            if self._expanded[i]:
                n = self._extra_points[i] + 1
                values = np.hstack((values[-n:-1], values, values[1:n]))
            hills.append(values)
        if len(self.bias_variables) == 1:
            bias = self.height*hills[0]
        else:
            bias = self.height*functools.reduce(np.multiply.outer, reversed(hills))
        self.add_bias(bias)
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
        extra_atom = f'ATOM      1  Cs   Cs A   1       0.000   0.000   0.000  1.00  0.00'
        pdb = app.PDBFile(io.StringIO(extra_atom))
        for i in range(len(self.variables)):
            modeller.add(pdb.topology, pdb.positions)
        num_particles = system.getNumParticles()
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
            parameter.setParticleParameters(0, num_particles+i, [])
        system.addForce(self.driving_force)

        super().__init__(modeller.topology, system, integrator, platform, platformProperties)
        self.context.setParameter('Lx', Vx.x)

    def add_periodic_task(self, task):
        task.initialize(self)
        self._periodic_tasks.append(task)

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
            Vx, Vy, _ = self.context.getState().getPeriodicBoxVectors()
            for i, v in enumerate(self.variables):
                y = Vy.y*(i + 1)/(len(self.variables) + 2)
                value = kwargs.get(v.id, v.colvar.evaluate(self.system, positions))
                position = v._particle_position(value, Vx.x, y)
                extended_positions.append(position*unit.nanometers)
            self.context.setPositions(extended_positions)

    def set_random_velocities(self, temperature, seed=None):
        """
        Sets the velocities of all particles in the simulation's context.

        .. warning ::
            The velocities of the extended-space variables are set to zero.

        Parameters
        ----------
            temperature : float or unit.Quantity
                The temperature.

        Keyword Args
        ------------
            seed : int, default=None
                A seed for the random number generator.

        """
        n = self.system.getNumParticles() - len(self.variables)
        masses = []
        for i, v in enumerate(self.variables):
            masses.append(self.system.getParticleMass(n+i))
            self.system.setParticleMass(n+i, 0)
        if seed is None:
            self.context.setVelocitiesToTemperature(temperature)
        else:
            self.context.setVelocitiesToTemperature(temperature, seed)
        for i, mass in enumerate(masses):
            self.system.setParticleMass(n+i, mass)

    def step(self, steps):
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
        driving_force : openmm.CustomCVForce
            The driving force.

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
        if dimension == 0 or height is None or frequency is None:
            self.metadynamics = None
        elif dimension > 3 or enforce_gridless:
            self.metadynamics = GridlessMetadynamics(variables, height, frequency, buffer_size)
        else:
            self.metadynamics = GriddedMetadynamics(variables, height, frequency, grid_expansion)

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

        if self.metadynamics is not None:
            simulation.add_periodic_task(self.metadynamics)

        if any(v.temperature != self.temperature for v in self.variables):
            simulation.context.setPositions([openmm.Vec3(0, 0, 0)]*system.getNumParticles())
            try:
                kT = integrator.getPerDofVariableByName('kT')
            except Exception:
                raise ValueError('Multiple temperatures require CustomIntegrator with per-dof variable `kT`')
            kB = _standardize(unit.MOLAR_GAS_CONSTANT_R)
            nparticles = system.getNumParticles() - len(self.variables)
            for i in range(nparticles):
                kT[i] = kB*self.temperature*openmm.Vec3(1, 1, 1)
            for i, v in enumerate(self.variables):
                kT[nparticles+i] = kB*v.temperature*openmm.Vec3(1, 0, 0)
            integrator.setPerDofVariableByName('kT', kT)

        return simulation
