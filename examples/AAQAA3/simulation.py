import argparse
import random
from simtk import openmm, unit
from simtk.openmm import app
from sys import stdout
import ufedmm
from ufedmm import cvlib

parser = argparse.ArgumentParser()
parser.add_argument('--seed', dest='seed', help='the RNG seed', default=None)
parser.add_argument('--platform', dest='platform', help='the computation platform', default='Reference')
args = parser.parse_args()

seed = random.SystemRandom().randint(0, 2**31) if args.seed is None else args.seed
temp = 300*unit.kelvin
gamma = 10/unit.picoseconds
dt = 2*unit.femtoseconds
nsteps = 1000000
mass = 30*unit.dalton*(unit.nanometer)**2
Ks = 1000*unit.kilojoules_per_mole
Ts = 1500*unit.kelvin
limit = 180*unit.degrees
sigma = 18*unit.degrees
height = 2.0*unit.kilojoules_per_mole
deposition_period = 200

pdb = app.PDBFile('aaqaa3.pdb')
force_field = app.ForceField('charmm36.xml')
system = force_field.createSystem(
    pdb.topology,
    nonbondedMethod=app.NoCutoff,
    constraints=None,
    rigidWater=False,
    removeCMMotion=False,
)


cv = ufedmm.CollectiveVariable('hc', cvlib.HelixRamachandranContent(pdb.topology, 0, 14))
s_hc = ufedmm.DynamicalVariable('s_hc', 0, 1, mass, Ts, cv, Ks, sigma=sigma, periodic=False)

ufed = ufedmm.UnifiedFreeEnergyDynamics([s_hc], temp, height, deposition_period)
ufedmm.serialize(ufed, 'ufed_object.yml')
integrator = ufedmm.GeodesicBAOABIntegrator(temp, gamma, dt)
integrator.setRandomNumberSeed(seed)
platform = openmm.Platform.getPlatformByName(args.platform)
simulation = ufed.simulation(pdb.topology, system, integrator, platform)
simulation.set_positions(pdb.positions)
simulation.minimizeEnergy(tolerance=0.1*unit.kilojoules_per_mole)
simulation.set_velocities_to_temperature(temp, seed)
output = ufedmm.MultipleFiles(stdout, 'output.csv')
reporter = ufedmm.StateDataReporter(output, 100, simulation.driving_force, step=True, speed=True)
simulation.reporters.append(reporter)
simulation.step(nsteps)
# 
# 
# 
# 
# 
# print(cv.evaluate(system, pdb.positions))
# 
# helix_context = cvlib.HelixHydrogenBondContent(pdb.topology, 0, 14)
# cv = ufedmm.CollectiveVariable('hc', helix_context)
# print(cv.evaluate(system, pdb.positions))
# 
# helix_context = cvlib.HelixAngleContent(pdb.topology, 0, 14)
# cv = ufedmm.CollectiveVariable('hc', helix_context)
# print(cv.evaluate(system, pdb.positions))
