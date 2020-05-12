"""
.. module:: ufedmm
   :platform: Unix, Windows
   :synopsis: Unified Free Energy Dynamics with OpenMM

.. moduleauthor:: Charlles Abreu <abreu@eq.ufrj.br>

.. _Context: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.Context.html
.. _CustomCVForce: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.CustomCVForce.html
.. _CustomIntegrator: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.CustomIntegrator.html
.. _Force: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.Force.html
.. _System: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.System.html

"""

import copy
import functools
import io

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

    """
    def __init__(self, id, force):
        if not id.isidentifier():
            raise ValueError('Parameter id must be a valid variable identifier')
        self.id = id
        self.force = force


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
        drive : float or unit.Quantity or str
            Either the value of the force constant of a harmonic driving force or an algebraic
            expression giving the energy of the system as a function of this dynamical variable and
            its associated collective variable. Such expression can also contain a set of global
            parameters, whose values must be passed as keyword arguments (see below).

    Keyword Args
    ------------
        sigma : float or unit.Quantity, default=None
            The standard deviation. If this is `None`, then no bias will be considered.
        grid_size : int, default=None
            The grid size. If this is `None` and `sigma` is finite, then a convenient value will be
            automatically chosen.
        periodic : bool, default=True
            Whether the collective variable is periodic with period `L=max_value-min_value`.
        **parameters
            Names and values of global parameters present in the algebraic expression defined as
            `drive` (see above).

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
    def __init__(self, id, min_value, max_value, mass, temperature, colvar, drive,
                 sigma=None, grid_size=None, periodic=True, **parameters):
        self.id = id
        self.colvar = colvar
        self.min_value = _standardize(min_value)
        self.max_value = _standardize(max_value)
        self._range = self.max_value - self.min_value
        self.mass = _standardize(mass)
        self.temperature = _standardize(temperature)
        self.sigma = _standardize(sigma)

        if sigma is None:
            self.grid_size = None
        else:
            self._scaled_variance = (self.sigma/self._range)**2
            if grid_size is None:
                self.grid_size = int(np.ceil(5*self._range/self.sigma)) + 1
            else:
                self.grid_size = grid_size
        self.periodic = periodic

        self.parameters = {}
        if isinstance(drive, str):
            self.drive = drive
            for key, value in parameters.items():
                self.parameters[key] = _standardize(value)
        else:
            if periodic:
                self.drive = f'0.5*K_{self.colvar.id}*min(d{self.colvar.id},{self._range}-d{self.colvar.id})^2'
                self.drive += f'; d{self.colvar.id}=abs({self.colvar.id}-{self.id})'
            else:
                self.drive = f'0.5*K_{self.colvar.id}*({self.colvar.id}-{self.id})^2'
            self.parameters[f'K_{self.colvar.id}'] = _standardize(drive)

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
            drive=self.drive,
            sigma=self.sigma,
            grid_size=self.grid_size,
            periodic=self.periodic,
            **self.parameters,
        )

    def __setstate__(self, kw):
        self.__init__(**kw)

    def _push_collective_variable(self, force):
        force.addCollectiveVariable(self.colvar.id, copy.deepcopy(self.colvar.force))
        for key, value in self.parameters.items():
            force.addGlobalParameter(key, value)

    def _push_extended_space_variable(self, force):
        if self.periodic:
            expression = f'{self.min_value}+{self._range}*(x/Lx-floor(x/Lx))'
        else:
            ramp_up = f'{self.min_value}+{2*self._range}*pos'
            ramp_down = f'{self.max_value+self._range}-{2*self._range}*pos'
            expression = f'select(step(0.5-pos),{ramp_up},{ramp_down})'
            expression += '; pos=x/Lx-floor(x/Lx)'
        parameter = openmm.CustomExternalForce(expression)
        parameter.addGlobalParameter('Lx', 0.0)
        parameter.addParticle(0, [])
        force.addCollectiveVariable(f'{self.id}', parameter)

    def _particle_mass(self, Lx):
        length = Lx if self.periodic else Lx/2
        return self.mass*(self._range/length)**2

    def _particle_position(self, value, Lx, y=0):
        length = Lx if self.periodic else Lx/2
        return openmm.Vec3(length*(value - self.min_value)/self._range, y, 0)

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
            >>> mass = 50*unit.dalton*(unit.nanometer/unit.radians)**2
            >>> K = 1000*unit.kilojoules_per_mole/unit.radians**2
            >>> Ts = 1500*unit.kelvin
            >>> bound = 180*unit.degrees
            >>> s_phi = ufedmm.DynamicalVariable('s_phi', -bound, bound, mass, Ts, model.phi, K)
            >>> s_psi = ufedmm.DynamicalVariable('s_psi', -bound, bound, mass, Ts, model.psi, K)
            >>> s_phi.evaluate(model.system, model.positions)
            3.141592653589793
            >>> s_psi.evaluate(model.system, model.positions)
            3.141592653589793

        """
        new_system = openmm.System()
        new_system.setDefaultPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
        for index in range(len(positions)):
            new_system.addParticle(system.getParticleMass(index))
        new_system.addForce(copy.deepcopy(self.colvar.force))
        platform = openmm.Platform.getPlatformByName('Reference')
        context = openmm.Context(new_system, openmm.CustomIntegrator(0), platform)
        context.setPositions(positions)
        energy = context.getState(getEnergy=True).getPotentialEnergy()
        return energy.value_in_unit(unit.kilojoules_per_mole)


class _Metadynamics(object):
    """
    Extended-space Metadynamics.

    """
    def __init__(self, variables, height, frequency, grid_expansion):
        self.bias_variables = [v for v in variables if v.sigma is not None]
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
            v._push_extended_space_variable(self.force)
        self.force.addTabulatedFunction('bias', self._table)

    def _add_bias(self, bias):
        self._bias += bias.flatten()
        if len(self.bias_variables) == 1:
            self._table.setFunctionParameters(self._bias, *self._bounds)
        else:
            self._table.setFunctionParameters(*self._widths, self._bias, *self._bounds)

    def _add_gaussian(self, position):
        gaussians = []
        for i, v in enumerate(self.bias_variables):
            x = (position[i] - v.min_value)/v._range
            if v.periodic:
                x = x % 1.0
            dist = np.abs(np.linspace(0, 1, num=v.grid_size) - x)
            if v.periodic:
                dist = np.min(np.array([dist, np.abs(dist-1)]), axis=0)
            values = np.exp(-0.5*dist*dist/v._scaled_variance)
            if self._expanded[i]:
                n = self._extra_points[i] + 1
                values = np.hstack((values[-n:-1], values, values[1:n]))
            gaussians.append(values)
        if len(self.bias_variables) == 1:
            bias = self.height*gaussians[0]
        else:
            bias = self.height*functools.reduce(np.multiply.outer, reversed(gaussians))
        self._add_bias(bias)

    def describeNextReport(self, simulation):
        steps = self.frequency - simulation.currentStep % self.frequency
        return (steps, False, False, False, False, False)

    def report(self, simulation, state):
        cv_values = self.force.getCollectiveVariableValues(simulation.context)
        self._add_gaussian(cv_values)
        self.force.updateParametersInContext(simulation.context)

    def update_bias_parameters(self, nparticles):
        for i, v in enumerate(self.bias_variables):
            parameter = self.force.getCollectiveVariable(i)
            parameter.setParticleParameters(0, nparticles+i, [])


class _Simulation(app.Simulation):
    def __init__(self, metadynamics, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metadynamics = metadynamics

    def step(self, steps):
        if self.metadynamics is None:
            self._simulate(endStep=self.currentStep+steps)
        else:
            self.reporters.append(self.metadynamics)
            self._simulate(endStep=self.currentStep+steps)
            self.reporters.pop()


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
        grid_expansion : int, default=20
            The grid expansion.
        height : float or unit.Quantity, default=None
            The height.
        frequency : int, default=None
            The frequency.

    Properties:
        driving_force : openmm.CustomCVForce
            The driving force.
        bias_force : openmm.CustomCVForce
            The bias force.

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
    def __init__(self, variables, temperature, height=None, frequency=None, grid_expansion=20):
        self.variables = variables
        self.temperature = _standardize(temperature)
        self.height = _standardize(height)
        self.frequency = frequency
        self.grid_expansion = grid_expansion

        self.driving_force = openmm.CustomCVForce(self.get_energy_function())
        for v in self.variables:
            v._push_collective_variable(self.driving_force)
            v._push_extended_space_variable(self.driving_force)

        if (all(v.sigma is None for v in self.variables) or height is None or frequency is None):
            self.bias_force = self._metadynamics = None
        else:
            self._metadynamics = _Metadynamics(self.variables, self.height, frequency, grid_expansion)
            self.bias_force = self._metadynamics.force

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
        )

    def __setstate__(self, kw):
        self.__init__(**kw)

    def get_energy_function(self):
        energies = [v.drive.split(';', 1) for v in self.variables]
        energy_terms, definitions = zip(*energies)
        expression = ';'.join(['+'.join(energy_terms)] + list(definitions))
        return expression

    def get_parameters(self):
        parameters = {}
        for v in self.variables:
            parameters.update(v.parameters)
        return parameters

    def set_positions(self, simulation, positions, extended=False):
        """
        Sets the positions of all particles in a simulation context.

        Parameters
        ----------
            simulation : openmm.Simulation
                The simulation.
            positions : list of openmm.Vec3
                The positions.

        """
        if extended:
            simulation.context.setPositions(positions)
        else:
            extended_positions = copy.deepcopy(positions)
            Vx, Vy, _ = simulation.context.getState().getPeriodicBoxVectors()
            for i, v in enumerate(self.variables):
                value = v.evaluate(simulation.system, positions)
                position = v._particle_position(value, Vx.x, y=Vy.y*(i+1)/(len(self.variables)+2))
                extended_positions.append(position*unit.nanometers)
            simulation.context.setPositions(extended_positions)

    def set_random_velocities(self, simulation, seed=None):
        """
        Sets the velocities of all particles in a simulation context.

        Parameters
        ----------
            simulation : openmm.Simulation
                The simulation.

        Keyword Args
        ------------
            seed : int, default=None
                A seed for the random number generator.

        """
        n = simulation.system.getNumParticles() - len(self.variables)
        masses = []
        for i, v in enumerate(self.variables):
            masses.append(simulation.system.getParticleMass(n+i))
            simulation.system.setParticleMass(n+i, 0)
        if seed is None:
            simulation.context.setVelocitiesToTemperature(self.temperature)
        else:
            simulation.context.setVelocitiesToTemperature(self.temperature, seed)
        for i, mass in enumerate(masses):
            simulation.system.setParticleMass(n+i, mass)

    def set_bias(self, simulation, bias):
        if simulation.metadynamics is not None:
            simulation.metadynamics.add_bias(bias)
            simulation.metadynamics.force.updateParametersInContext(simulation.context)

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
        for force in system.getForces():
            cls = force.__class__.__name__
            if cls == 'CMMotionRemover' or cls.endswith('Barostat'):
                raise Exception('UFED: system cannot contain CMMotionRemover nor any Barostat')

        Vx, Vy, Vz = topology.getPeriodicBoxVectors()
        if not (Vx.y == Vx.z == Vy.x == Vy.z == Vz.x == Vz.y == 0.0):
            raise ValueError('UFED: only orthorhombic boxes are allowed')

        positions = [openmm.Vec3(0, 0, 0) for atom in topology.atoms()]
        modeller = app.Modeller(topology, positions)
        extra_atom = f'ATOM      1  Cs   Cs A   1       0.000   0.000   0.000  1.00  0.00'
        pdb = app.PDBFile(io.StringIO(extra_atom))
        for i in range(len(self.variables)):
            modeller.add(pdb.topology, pdb.positions)
        nparticles = system.getNumParticles()
        nb_types = (openmm.NonbondedForce, openmm.CustomNonbondedForce)
        nb_forces = [f for f in system.getForces() if isinstance(f, nb_types)]
        for i, v in enumerate(self.variables):
            system.addParticle(v._particle_mass(Vx.x))
            for nb_force in nb_forces:
                if isinstance(nb_force, openmm.NonbondedForce):
                    nb_force.addParticle(0.0, 1.0, 0.0)
                else:
                    nb_force.addParticle([0.0]*nb_force.getNumPerParticleParameters())
            parameter = self.driving_force.getCollectiveVariable(2*i+1)
            parameter.setParticleParameters(0, nparticles+i, [])
        system.addForce(self.driving_force)

        if self._metadynamics:
            self._metadynamics.update_bias_parameters(nparticles)
            system.addForce(self.bias_force)

        simulation = _Simulation(
            self._metadynamics,
            modeller.topology,
            system,
            integrator,
            platform,
            platformProperties,
        )
        simulation.context.setParameter('Lx', Vx.x)

        if any(v.temperature != self.temperature for v in self.variables):
            simulation.context.setPositions(modeller.positions)
            try:
                kT = integrator.getPerDofVariableByName('kT')
            except Exception:
                raise ValueError('Multiple temperatures require CustomIntegrator with per-dof variable `kT`')
            kB = _standardize(unit.MOLAR_GAS_CONSTANT_R)
            vec3 = openmm.Vec3(1, 1, 1)
            for i in range(nparticles):
                kT[i] = kB*self.temperature*vec3
            for i, v in enumerate(self.variables):
                kT[nparticles+i] = kB*v.temperature*vec3
            integrator.setPerDofVariableByName('kT', kT)

        return simulation
