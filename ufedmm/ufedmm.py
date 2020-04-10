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
                 sigma=0, grid_size=None, periodic=True):
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
        self.grid_size = grid_size
        self.periodic = periodic
        self._range = self.max_value - self.min_value
        self._scaled_variance = (self.sigma/self._range)**2

    def __repr__(self):
        properties = f'm={self.mass}, K={self.force_constant}, T={self.temperature}'
        return f'<{self.id} in [{self.min_value}, {self.max_value}], {properties}>'

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
            >>> psi = ufedmm.CollectiveVariable('psi', model.psi, -180*unit.degrees, 180*unit.degrees, mass, K, Ts)
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
        system : openmm.System
            The system.
        topology : openmm.app.Topology
            The topology.
        positions : list of openmm.Vec3
            The positions.
        temperature : float or unit.Quantity
            The temperature.
        height : float or unit.Quantity
            The height.
        frequency : int
            The frequency.

    Keyword Args
    ------------
        grid_expansion : int, default=20
            The grid expansion.

    """
    def __init__(self, variables, system, topology, positions, temperature, height, frequency, grid_expansion=20):
        self.system = copy.deepcopy(system)
        self.variables = variables
        self._modeller = app.Modeller(topology, positions)
        self._temperature = _standardize(temperature)
        self._height = _standardize(height)
        self._frequency = frequency
        Vx, Vy, Vz = self._modeller.topology.getPeriodicBoxVectors()
        if not (Vx.y == Vx.z == Vy.x == Vy.z == Vz.x == Vz.y == 0.0):
            raise ValueError('Only orthorhombic boxes are allowed')
        self._Lx = Vx.x
        nbforce = [f for f in self.system.getForces() if isinstance(f, openmm.NonbondedForce)][0]
        energy_terms = []
        definitions = []
        for i, cv in enumerate(self.variables):
            value = cv.evaluate(self.positions)
            new_atom = self._new_atom(x=self._Lx*(value - cv.min_value)/cv._range, y=i)
            self._modeller.add(*new_atom)
            self.system.addParticle(cv.mass*(cv._range/self._Lx)**2)
            nbforce.addParticle(0.0, 1.0, 0.0)
            energy_terms.append(f'0.5*K_{cv.id}*min(d{cv.id},{cv._range}-d{cv.id})^2')
            definitions.append(f'd{cv.id}=abs({cv.id}-s_{cv.id})')
        parameter_list = ', '.join(f's_{v.id}' for v in self.variables)
        energy_terms.append(f'bias({parameter_list})')
        expression = '; '.join([' + '.join(energy_terms)] + definitions)
        force = self.force = openmm.CustomCVForce(expression)
        for cv in self.variables:
            force.addGlobalParameter(f'K_{cv.id}', cv.force_constant)
            force.addCollectiveVariable(cv.id, cv.force)
        n = self.system.getNumParticles() - len(self.variables)
        self._widths = []
        self._bounds = []
        for i, cv in enumerate(self.variables):
            expression = f'{cv.min_value}+{cv._range}*(x-floor(x)); x=x1/{self._Lx}'
            parameter = openmm.CustomCompoundBondForce(1, expression)
            parameter.addBond([n+i], [])
            force.addCollectiveVariable(f's_{cv.id}', parameter)
            cv._expanded = cv.periodic and len(self.variables) > 1
            cv._extra_points = min(grid_expansion, cv.grid_size) if cv._expanded else 0
            extra_range = cv._extra_points*cv._range/(cv.grid_size - 1)
            self._widths += [cv.grid_size + 2*cv._extra_points]
            self._bounds += [cv.min_value - extra_range, cv.max_value + extra_range]
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
        force.addTabulatedFunction('bias', self._table)
        self.system.addForce(force)

    def _new_atom(self, x=0, y=0, z=0):
        xa, ya, za = 10*x, 10*y, 10*z
        pdb = app.PDBFile(io.StringIO(
            f'ATOM      1  Cl   Cl A   1     {xa:7.3f} {ya:7.3f} {za:7.3f}  1.00  0.00'
        ))
        return pdb.topology, pdb.positions

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
            if cv._expanded:
                n = cv._extra_points + 1
                values = np.hstack((values[-n:-1], values, values[1:n]))
            gaussians.append(values)

        if len(self.variables) == 1:
            self._bias += self._height*gaussians[0]
            periodic = self.variables[0].periodic
            self._table.setFunctionParameters(self._bias.flatten(), *self._bounds, periodic)
        else:
            self._bias += self._height*functools.reduce(np.multiply.outer, reversed(gaussians))
            self._table.setFunctionParameters(*self._widths, self._bias.flatten(), *self._bounds)

    def simulation(self, integrator, platform=None, properties=None, seed=None):
        """
        Returns a Simulation.

        Parameters
        ----------
            integrator :
                The integrator. If the temperature of any collective variable is different from
                the system temperature, then this must be a CustomIntegrator with a per-dof variable
                called `kT`.

        Keyword Args
        ------------
            platform : openmm.Platform, default=None
                The platform.
            properties : dict, default=None
                The platform properties.
            seed : int, default=None
                The random number generator seed.

        """

        simulation = openmm.app.Simulation(self.topology, self.system, integrator, platform, properties)
        simulation.context.setPositions(self.positions)
        if seed is None:
            simulation.context.setVelocitiesToTemperature(self._temperature)
        else:
            simulation.context.setVelocitiesToTemperature(self._temperature, seed)
        n = self.system.getNumParticles() - len(self.variables)
        kB = _standardize(unit.MOLAR_GAS_CONSTANT_R)
        kT = [kB*self._temperature*openmm.Vec3(1, 1, 1) for i in range(n)]
        for cv in self.variables:
            kT.append(kB*cv.temperature*openmm.Vec3(1, 0, 0))
        integrator.setPerDofVariableByName('kT', kT)
        simulation.reporters.append(self)
        return simulation

    @property
    def topology(self):
        return self._modeller.topology

    @property
    def positions(self):
        return self._modeller.positions

    def describeNextReport(self, simulation):
        steps = self._frequency - simulation.currentStep % self._frequency
        return (steps, False, False, False, False, False)

    def report(self, simulation, state):
        cv_values = self.force.getCollectiveVariableValues(simulation.context)
        position = cv_values[len(self.variables):]
        self._add_gaussian(position)
        self.force.updateParametersInContext(simulation.context)
