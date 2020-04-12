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
        force : openmm.Force
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
        sigma : float or unit.Quantity, default=0
            The standard deviation. If this is `0`, then no gaussians will be deposited.
        grid_size : int, default=None
            The grid size. If this is `None` and `sigma` is finite, then a convenient value will be
            automatically chosen.
        periodic : bool, default=True
            In the current version, only periodic variables are permitted.

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
    def __init__(self, id, force, min_value, max_value, mass, force_constant, temperature,
                 sigma=0.0, grid_size=None, periodic=True):
        if not id.isidentifier():
            raise ValueError('Parameter id must be a valid variable identifier')
        if not periodic:
            raise ValueError('UFED currently requires periodic variables')
        self.id = id
        self.force = force
        self.min_value = _standardize(min_value)
        self.max_value = _standardize(max_value)
        self.mass = _standardize(mass)
        self.force_constant = _standardize(force_constant)
        self.temperature = _standardize(temperature)
        self.sigma = _standardize(sigma)
        if grid_size is None and sigma != 0.0:
            self.grid_size = int(np.ceil(5*self._range/self.sigma))
        else:
            self.grid_size = grid_size
        self.periodic = periodic
        self._range = self.max_value - self.min_value
        self._scaled_variance = (self.sigma/self._range)**2

    def __repr__(self):
        properties = f'm={self.mass}, K={self.force_constant}, T={self.temperature}'
        return f'<{self.id} in [{self.min_value}, {self.max_value}], {properties}>'

    def __getstate__(self):
        return dict(
            id=self.id,
            force=self.force,
            min_value=self.min_value,
            max_value=self.max_value,
            mass=self.mass,
            force_constant=self.force_constant,
            temperature=self.temperature,
            sigma=self.sigma,
            grid_size=self.grid_size,
            periodic=self.periodic,
        )

    def __setstate__(self, kw):
        self.__init__(**kw)

    def evaluate(self, positions, box_vectors=None):
        """
        Computes the value of the collective variable for a given set of particle coordinates
        and box vectors. Whether periodic boundary conditions will be used or not depends on
        the corresponding attribute of the Force_ object specified as the collective variable.

        Parameters
        ----------
            positions : list of openmm.Vec3
                A list whose length equals the number of particles in the system and which contains
                the coordinates of these particles.

        Keyword Args
        ------------
            box_vectors : list of openmm.Vec3, default=None
                A list with three vectors which describe the edges of the simulation box.

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
            >>> phi.evaluate(model.positions)
            3.141592653589793
            >>> psi.evaluate(model.positions)
            3.141592653589793

        """
        system = openmm.System()
        for position in positions:
            system.addParticle(0)
        if box_vectors is not None:
            system.setDefaultPeriodicBoxVectors(*box_vectors)
        system.addForce(copy.deepcopy(self.force))
        platform = openmm.Platform.getPlatformByName('Reference')
        context = openmm.Context(system, openmm.CustomIntegrator(0), platform)
        context.setPositions(positions)
        energy = context.getState(getEnergy=True).getPotentialEnergy()
        return energy.value_in_unit(unit.kilojoules_per_mole)


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
        self._metadynamics = (any(cv.sigma != 0.0 for cv in self.variables)
                              and height is not None and frequency is not None)

        energy_terms = []
        definitions = []
        for i, cv in enumerate(self.variables):
            energy_terms.append(f'0.5*K_{cv.id}*min(d{cv.id},{cv._range}-d{cv.id})^2')
            definitions.append(f'd{cv.id}=abs({cv.id}-s_{cv.id})')
        expression = '; '.join([' + '.join(energy_terms)] + definitions)
        self.force = openmm.CustomCVForce(expression)
        parameters = {}
        for i, cv in enumerate(self.variables):
            self.force.addGlobalParameter(f'K_{cv.id}', cv.force_constant)
            self.force.addCollectiveVariable(cv.id, cv.force)
            expression = f'{cv.min_value}+{cv._range}*(x-floor(x)); x=x1/Lx'
            parameter = openmm.CustomCompoundBondForce(1, expression)
            parameter.addGlobalParameter('Lx', 0.0)
            parameter.addBond([0], [])
            self.force.addCollectiveVariable(f's_{cv.id}', parameter)
            parameters[f's_{cv.id}'] = parameter

        if self._metadynamics:
            self.bias_force = openmm.CustomCVForce('bias({})'.format(', '.join(parameters)))
            for id, parameter in parameters.items():
                self.bias_force.addCollectiveVariable(id, copy.deepcopy(parameter))
            self._widths = []
            self._bounds = []
            self._expanded = []
            self._extra_points = []
            for cv in self.variables:
                expanded = cv.periodic and len(self.variables) > 1
                extra_points = min(grid_expansion, cv.grid_size) if expanded else 0
                extra_range = extra_points*cv._range/(cv.grid_size - 1)
                self._widths += [cv.grid_size + 2*extra_points]
                self._bounds += [cv.min_value - extra_range, cv.max_value + extra_range]
                self._expanded += [expanded]
                self._extra_points += [extra_points]
            self._bias = np.zeros(tuple(reversed(self._widths)))
            if len(variables) == 1:
                periodic = self.variables[0].periodic
                self._table = openmm.Continuous1DFunction(self._bias.flatten(), *self._bounds, periodic)
            elif len(variables) == 2:
                self._table = openmm.Continuous2DFunction(*self._widths, self._bias.flatten(), *self._bounds)
            elif len(variables) == 3:
                self._table = openmm.Continuous3DFunction(*self._widths, self._bias.flatten(), *self._bounds)
            else:
                raise ValueError('UFED requires 1, 2, or 3 collective variables')
            self.bias_force.addTabulatedFunction('bias', self._table)

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

    def _add_gaussian(self, position):
        gaussians = []
        for i, cv in enumerate(self.variables):
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
        if len(self.variables) == 1:
            self._bias += self.height*gaussians[0]
            periodic = self.variables[0].periodic
            self._table.setFunctionParameters(self._bias.flatten(), *self._bounds, periodic)
        else:
            self._bias += self.height*functools.reduce(np.multiply.outer, reversed(gaussians))
            self._table.setFunctionParameters(*self._widths, self._bias.flatten(), *self._bounds)

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
            value = cv.evaluate(positions)
            position = openmm.Vec3(Lx*(value - cv.min_value)/cv._range, i, 0)
            extended_positions.append(position*unit.nanometers)
        simulation.context.setPositions(extended_positions)

    def simulation(self, topology, system, integrator, platform=None, platformProperties=None):
        """
        Returns a Simulation.

        Parameters
        ----------
            topology : openmm.app.Topology
                The topology.
            system : openmm.System
                The system.
            integrator :
                The integrator. If the temperature of any collective variable is different from
                the system temperature, then this must be a CustomIntegrator with a per-dof variable
                called `kT`.

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
            >>> integrator = ufedmm.GeodesicBAOABIntegrator(dt, 300*unit.kelvin, gamma)
            >>> simulation = ufed.simulation(model.topology, model.system, integrator)

        """
        Vx, Vy, Vz = topology.getPeriodicBoxVectors()
        if not (Vx.y == Vx.z == Vy.x == Vy.z == Vz.x == Vz.y == 0.0):
            raise ValueError('UFED: only orthorhombic boxes are allowed')
        Lx = Vx.x

        positions = [openmm.Vec3(0, 0, 0) for atom in topology.atoms()]
        modeller = app.Modeller(topology, positions)
        for y, cv in enumerate(self.variables):
            new_atom = app.PDBFile(io.StringIO(
                f'ATOM      1  Cl   Cl A   1       0.000 {y:3d}.000   0.000  1.00  0.00'
            ))
            modeller.add(new_atom.topology, new_atom.positions)

        internal_system = copy.deepcopy(system)
        nparticles = internal_system.getNumParticles()
        force = copy.deepcopy(self.force)
        nb_types = (openmm.NonbondedForce, openmm.CustomNonbondedForce)
        nb_forces = [f for f in internal_system.getForces() if isinstance(f, nb_types)]
        for i, cv in enumerate(self.variables):
            internal_system.addParticle(cv.mass*(cv._range/Lx)**2)
            for nb_force in nb_forces:
                if isinstance(nb_force, openmm.NonbondedForce):
                    nb_force.addParticle(0.0, 1.0, 0.0)
                else:
                    nb_force.addParticle([0.0]*nb_force.getNumPerParticleParameters())
            force.getCollectiveVariable(2*i+1).setBondParameters(0, [nparticles+i], [])
        internal_system.addForce(force)

        if self._metadynamics:
            internal_system.addForce(self.bias_force)

        simulation = openmm.app.Simulation(
            modeller.topology, internal_system, integrator, platform, platformProperties,
        )
        simulation.force = force
        simulation.context.setPositions(modeller.positions)
        simulation.context.setParameter('Lx', Lx)
        simulation.context.setVelocitiesToTemperature(self.temperature)

        if any(cv.temperature != self.temperature for cv in self.variables):
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

        if self._metadynamics:
            simulation.reporters.append(self)
        return simulation

    def describeNextReport(self, simulation):
        steps = self.frequency - simulation.currentStep % self.frequency
        return (steps, False, False, False, False, False)

    def report(self, simulation, state):
        cv_values = self.bias_force.getCollectiveVariableValues(simulation.context)
        self._add_gaussian(cv_values)
        self.bias_force.updateParametersInContext(simulation.context)
