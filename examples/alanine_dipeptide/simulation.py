import argparse
import random
import ufedmm

from simtk import openmm, unit
from sys import stdout

waters = ['spce', 'tip3p', 'tip4pew', 'tip5p']
parser = argparse.ArgumentParser()
parser.add_argument('--water', dest='water', help='the water model', choices=waters, default=None)
parser.add_argument('--ff', dest='ff', help='the pepdide force field', default='amber03')
parser.add_argument('--seed', dest='seed', help='the RNG seed', default=None)
parser.add_argument('--platform', dest='platform', help='the computation platform', default='Reference')
args = parser.parse_args()

seed = random.SystemRandom().randint(0, 2**31) if args.seed is None else args.seed
model = ufedmm.AlanineDipeptideModel(force_field=args.ff, water=args.water)
temp = 300*unit.kelvin
gamma = 10/unit.picoseconds
dt = 2*unit.femtoseconds
nsteps = 1000000
mass = 30*unit.dalton*(unit.nanometer/unit.radians)**2
Ks = 1000*unit.kilojoules_per_mole/unit.radians**2
Ts = 1500*unit.kelvin
limit = 180*unit.degrees
sigma = 18*unit.degrees
height = 2.0*unit.kilojoules_per_mole
deposition_period = 200
s_phi = ufedmm.DynamicalVariable('s_phi', -limit, limit, mass, Ts, model.phi, Ks, sigma=sigma)
s_psi = ufedmm.DynamicalVariable('s_psi', -limit, limit, mass, Ts, model.psi, Ks, sigma=sigma)
ufed = ufedmm.UnifiedFreeEnergyDynamics([s_phi, s_psi], temp, height, deposition_period)
ufedmm.serialize(ufed, 'ufed_object.yml')
integrator = ufedmm.GeodesicBAOABIntegrator(temp, gamma, dt)
integrator.setRandomNumberSeed(seed)
platform = openmm.Platform.getPlatformByName(args.platform)
simulation = ufed.simulation(model.topology, model.system, integrator, platform)
simulation.context.setPositions(model.positions)
simulation.context.setVelocitiesToTemperature(temp, seed)
output = ufedmm.Tee(stdout, 'output.csv')
reporter = ufedmm.StateDataReporter(output, 100, step=True, speed=True, variables=True)
simulation.reporters.append(reporter)
simulation.step(nsteps)
