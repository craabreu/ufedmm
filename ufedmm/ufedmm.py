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
    A collective variable whose dynamics is meant to be driven by an extended-space variable.

    Parameters
    ----------
        id : str
            A valid identifier string for this collective variable.
        openmm_force : openmm.Force
            An OpenMM Force_ object whose energy function is used to evaluate the collective
            variable for a given Context_.
        min_value : float or unit.Quantity
            The minimum value.
        max_value : float or unit.Quantity
            The maximum value.
        mass : float or unit.Quantity
            The minimum value.
        force_constant : float or unit.Quantity
            The force constant.
        temperature : float or unit.Quantity
            The temperature.

    Keyword Args
    ------------
        sigma : float or unit.Quantity, default=None
            The standard deviation. If this is `None`, then no bias will be considered.
        grid_size : int, default=None
            The grid size. If this is `None` and `sigma` is finite, then a convenient value will be
            automatically chosen.
        repulsion_length : float or unit.Quantity, default=0.0
            Distance from each boundary (either `min_value` or `max_value`) at which the parameter
            that drives this collective variable begins to be repelled by a Weeks-Chandler-Andersen
            potential. If `repulsion_length=0` (default), then periodic boundary conditions will be
            considered.

    Example
    -------
        >>> import ufedmm
        >>> from simtk import openmm, unit
        >>> cv = openmm.CustomTorsionForce('theta')
        >>> cv.addTorsion(0, 1, 2, 3, [])
        0
        >>> mass = 50*unit.dalton*(unit.nanometer/unit.radians)**2
        >>> K = 1000*unit.kilojoules_per_mole/unit.radians**2
        >>> Ts = 1500*unit.kelvin
        >>> psi = ufedmm.CollectiveVariable('psi', cv, -180*unit.degrees, 180*unit.degrees, mass, K, Ts)
        >>> print(psi)
        <psi in [-3.141592653589793, 3.141592653589793], m=50, K=1000, T=1500>

    """
    def __init__(self, id, openmm_force, min_value, max_value, mass, force_constant, temperature,
                 sigma=None, grid_size=None, repulsion_length=0.0):
        if not id.isidentifier():
            raise ValueError('Parameter id must be a valid variable identifier')
        self.id = id
        self.openmm_force = openmm_force
        self.min_value = _standardize(min_value)
        self.max_value = _standardize(max_value)
        self._range = self.max_value - self.min_value
        self.mass = _standardize(mass)
        self.force_constant = _standardize(force_constant)
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
        self.repulsion_length = _standardize(repulsion_length)
        self.periodic = self.repulsion_length == 0.0

    def __repr__(self):
        properties = f'm={self.mass}, K={self.force_constant}, T={self.temperature}'
        return f'<{self.id} in [{self.min_value}, {self.max_value}], {properties}>'

    def __getstate__(self):
        return dict(
            id=self.id,
            openmm_force=self.openmm_force,
            min_value=self.min_value,
            max_value=self.max_value,
            mass=self.mass,
            force_constant=self.force_constant,
            temperature=self.temperature,
            sigma=self.sigma,
            grid_size=self.grid_size,
            repulsion_length=self.repulsion_length,
        )

    def __setstate__(self, kw):
        self.__init__(**kw)

    def evaluate(self, system, positions):
        """
        Computes the value of the collective variable for a given set of particle coordinates
        and box vectors. Whether periodic boundary conditions will be used or not depends on
        the corresponding attribute of the Force_ object specified as the collective variable.

        Parameters
        ----------
            system : openmm.System
                The system.
            positions : list of openmm.Vec3
                A list whose length equals the number of particles in the system and which contains
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
            >>> phi = ufedmm.CollectiveVariable('phi', model.phi, -bound, bound, mass, K, Ts)
            >>> psi = ufedmm.CollectiveVariable('psi', model.psi, -bound, bound, mass, K, Ts)
            >>> phi.evaluate(model.system, model.positions)
            3.141592653589793
            >>> psi.evaluate(model.system, model.positions)
            3.141592653589793

        """
        new_system = openmm.System()
        new_system.setDefaultPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
        for index in range(len(positions)):
            new_system.addParticle(system.getParticleMass(index))
        new_system.addForce(copy.deepcopy(self.openmm_force))
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
        self.bias_variables = [cv for cv in variables if cv.sigma is not None]
        self.height = height
        self.frequency = frequency
        self.grid_expansion = grid_expansion
        self._widths = []
        self._bounds = []
        self._expanded = []
        self._extra_points = []
        for cv in self.bias_variables:
            expanded = cv.periodic  # and len(self.bias_variables) > 1
            extra_points = min(grid_expansion, cv.grid_size) if expanded else 0
            extra_range = extra_points*cv._range/(cv.grid_size - 1)
            self._widths += [cv.grid_size + 2*extra_points]
            self._bounds += [cv.min_value - extra_range, cv.max_value + extra_range]
            self._expanded += [expanded]
            self._extra_points += [extra_points]
        self._bias = np.zeros(tuple(reversed(self._widths)))
        if len(variables) == 1:
            self._table = openmm.Continuous1DFunction(
                self._bias.flatten(),
                *self._bounds,
                # self.bias_variables[0].periodic,
            )
        elif len(variables) == 2:
            self._table = openmm.Continuous2DFunction(
                *self._widths,
                self._bias.flatten(),
                *self._bounds,
            )
        elif len(variables) == 3:
            self._table = openmm.Continuous3DFunction(
                *self._widths,
                self._bias.flatten(),
                *self._bounds,
            )
        else:
            raise ValueError('UFED requires 1, 2, or 3 biased collective variables')
        parameter_list = ', '.join(f's_{cv.id}' for cv in self.bias_variables)
        self.force = openmm.CustomCVForce(f'bias({parameter_list})')
        for cv in self.bias_variables:
            if cv.periodic:
                expression = f'{cv.min_value}+{cv._range}*(x/Lx-floor(x/Lx))'
            else:
                expression = f'{cv.min_value}+{cv._range}*(2*x/Lx+1/2)'
            parameter = openmm.CustomExternalForce(expression)
            parameter.addGlobalParameter('Lx', 0.0)
            parameter.addParticle(0, [])
            self.force.addCollectiveVariable(f's_{cv.id}', parameter)
        self.force.addTabulatedFunction('bias', self._table)

    def _add_gaussian(self, position):
        gaussians = []
        for i, cv in enumerate(self.bias_variables):
            x = (position[i] - cv.min_value)/cv._range
            if cv.periodic:
                x = x % 1.0
            dist = np.abs(np.linspace(0, 1, num=cv.grid_size) - x)
            if cv.periodic:
                dist = np.min(np.array([dist, np.abs(dist-1)]), axis=0)
            values = np.exp(-0.5*dist*dist/cv._scaled_variance)
            if self._expanded[i]:
                n = self._extra_points[i] + 1
                values = np.hstack((values[-n:-1], values, values[1:n]))
            gaussians.append(values)
        if len(self.bias_variables) == 1:
            self._bias += self.height*gaussians[0]
            self._table.setFunctionParameters(self._bias.flatten(), *self._bounds)
        else:
            self._bias += self.height*functools.reduce(np.multiply.outer, reversed(gaussians))
            self._table.setFunctionParameters(*self._widths, self._bias.flatten(), *self._bounds)

    def describeNextReport(self, simulation):
        steps = self.frequency - simulation.currentStep % self.frequency
        return (steps, False, False, False, False, False)

    def report(self, simulation, state):
        cv_values = self.force.getCollectiveVariableValues(simulation.context)
        self._add_gaussian(cv_values)
        self.force.updateParametersInContext(simulation.context)

    def update_bias_parameters(self, nparticles):
        for i, cv in enumerate(self.bias_variables):
            parameter = self.force.getCollectiveVariable(i)
            parameter.setParticleParameters(0, nparticles+i, [])


class _Simulation(app.Simulation):
    def __init__(self, metadynamics, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._metadynamics = metadynamics

    def step(self, steps):
        if self._metadynamics is None:
            self._simulate(endStep=self.currentStep+steps)
        else:
            self.reporters.append(self._metadynamics)
            self._simulate(endStep=self.currentStep+steps)
            self.reporters.pop()


class UnifiedFreeEnergyDynamics(object):
    """
    A Unified Free-Energy Dynamics (UFED) setup.

    Parameters
    ----------
        variables : list of CollectiveVariable
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
        >>> phi = ufedmm.CollectiveVariable('phi', model.phi, -limit, limit, mass, Ks, Ts)
        >>> psi = ufedmm.CollectiveVariable('psi', model.psi, -limit, limit, mass, Ks, Ts)
        >>> ufedmm.UnifiedFreeEnergyDynamics([phi, psi], 300*unit.kelvin)
        <variables=[phi, psi], temperature=300, height=None, frequency=None>

    """
    def __init__(self, variables, temperature, height=None, frequency=None, grid_expansion=20):
        self.variables = variables
        self.temperature = _standardize(temperature)
        self.height = _standardize(height)
        self.frequency = frequency
        self.grid_expansion = grid_expansion

        energy_terms = []
        definitions = []
        for i, cv in enumerate(self.variables):
            if cv.periodic:
                energy_terms.append(f'0.5*K_{cv.id}*min(d{cv.id},{cv._range}-d{cv.id})^2')
                definitions.append(f'd{cv.id}=abs({cv.id}-s_{cv.id})')
            else:
                driving = f'0.5*K_{cv.id}*({cv.id}-s_{cv.id})^2'
                repulsion_from_min = f'step(xmin-1)*(xmin^12-2*xmin^6+1)'
                repulsion_from_max = f'step(xmax-1)*(xmax^12-2*xmax^6+1)'
                energy_terms.append(f'{driving}+{repulsion_from_min}+{repulsion_from_max}')
                definitions.append(f'xmin={cv.repulsion_length}/(s_{cv.id}-{cv.min_value})')
                definitions.append(f'xmax={cv.repulsion_length}/({cv.max_value}-s_{cv.id})')
        expression = '; '.join([' + '.join(energy_terms)] + definitions)
        self.driving_force = openmm.CustomCVForce(expression)
        for i, cv in enumerate(self.variables):
            self.driving_force.addGlobalParameter(f'K_{cv.id}', cv.force_constant)
            self.driving_force.addCollectiveVariable(cv.id, cv.openmm_force)
            if cv.periodic:
                expression = f'{cv.min_value}+{cv._range}*(x/Lx-floor(x/Lx))'
            else:
                expression = f'{cv.min_value}+{cv._range}*(2*x/Lx+1/2)'
            parameter = openmm.CustomExternalForce(expression)
            parameter.addGlobalParameter('Lx', 0.0)
            parameter.addParticle(0, [])
            self.driving_force.addCollectiveVariable(f's_{cv.id}', parameter)

        if (all(cv.sigma is None for cv in self.variables) or height is None or frequency is None):
            self.bias_force = self._metadynamics = None
        else:
            self._metadynamics = _Metadynamics(
                self.variables,
                self.height,
                frequency,
                grid_expansion,
            )
            self.bias_force = self._metadynamics.force

    def __repr__(self):
        properties = f'temperature={self.temperature}, height={self.height}, frequency={self.frequency}'
        return f'<variables=[{", ".join(cv.id for cv in self.variables)}], {properties}>'

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

    def set_positions(self, simulation, positions):
        """
        Sets the positions of all particles in a simulation context.

        Parameters
        ----------
            simulation : openmm.Simulation
                The simulation.
            positions : list of openmm.Vec3
                The positions.

        """
        extended_positions = copy.deepcopy(positions)
        Lx = simulation.context.getParameter('Lx')
        for i, cv in enumerate(self.variables):
            value = cv.evaluate(simulation.system, positions)
            if cv.periodic:
                position = openmm.Vec3(Lx*(value - cv.min_value)/cv._range, i, 0)
            else:
                position = openmm.Vec3((Lx/2.0)*(((value - cv.min_value)/cv._range) - 0.5), i, 0)
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
        for i, cv in enumerate(self.variables):
            masses.append(simulation.system.getParticleMass(n+i))
            simulation.system.setParticleMass(n+i, 0)
        if seed is None:
            simulation.context.setVelocitiesToTemperature(self.temperature)
        else:
            simulation.context.setVelocitiesToTemperature(self.temperature, seed)
        for i, mass in enumerate(masses):
            simulation.system.setParticleMass(n+i, mass)

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
            >>> phi = ufedmm.CollectiveVariable('phi', model.phi, -limit, limit, mass, Ks, Ts)
            >>> psi = ufedmm.CollectiveVariable('psi', model.psi, -limit, limit, mass, Ks, Ts)
            >>> ufed = ufedmm.UnifiedFreeEnergyDynamics([phi, psi], 300*unit.kelvin)
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
        Lx = Vx.x

        positions = [openmm.Vec3(0, 0, 0) for atom in topology.atoms()]
        modeller = app.Modeller(topology, positions)
        for y, cv in enumerate(self.variables):
            new_atom = app.PDBFile(io.StringIO(
                f'ATOM      1  Cs   Cs A   1       0.000 {y:3d}.000   0.000  1.00  0.00'
            ))
            modeller.add(new_atom.topology, new_atom.positions)

        nparticles = system.getNumParticles()
        nb_types = (openmm.NonbondedForce, openmm.CustomNonbondedForce)
        nb_forces = [f for f in system.getForces() if isinstance(f, nb_types)]
        for i, cv in enumerate(self.variables):
            if cv.periodic:
                system.addParticle(cv.mass*(cv._range/Lx)**2)
            else:
                system.addParticle(cv.mass*(2*cv._range/Lx)**2)
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
        simulation.context.setParameter('Lx', Lx)

        if any(cv.temperature != self.temperature for cv in self.variables):
            simulation.context.setPositions(modeller.positions)
            try:
                kT = integrator.getPerDofVariableByName('kT')
            except Exception:
                raise ValueError('Multiple temperatures require CustomIntegrator with per-dof variable `kT`')
            kB = _standardize(unit.MOLAR_GAS_CONSTANT_R)
            vec3 = openmm.Vec3(1, 1, 1)
            for i in range(nparticles):
                kT[i] = kB*self.temperature*vec3
            for i, cv in enumerate(self.variables):
                kT[nparticles+i] = kB*cv.temperature*vec3
            integrator.setPerDofVariableByName('kT', kT)

        return simulation
