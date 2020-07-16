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
mass = 1000*unit.dalton*(unit.nanometer)**2
Ks = 5000*unit.kilojoules_per_mole
Ts = 3000*unit.kelvin
sigma = 0.05
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

# cv = ufedmm.CollectiveVariable('hc', cvlib.HelixHydrogenBondContent(pdb.topology, 1, 13))
# cv = ufedmm.CollectiveVariable('hc', cvlib.HelixAngleContent(pdb.topology, 0, 14))
cv = ufedmm.CollectiveVariable('hc', cvlib.HelixRamachandranContent(pdb.topology, 0, 14))

s_hc = ufedmm.DynamicalVariable('s_hc', 0.0, 1.0, mass, Ts, cv, Ks, sigma=sigma, periodic=False)

ufed = ufedmm.UnifiedFreeEnergyDynamics([s_hc], temp, height, deposition_period)
ufedmm.serialize(ufed, 'ufed_object.yml')
integrator = ufedmm.GeodesicBAOABIntegrator(temp, gamma, dt)
integrator.setRandomNumberSeed(seed)
platform = openmm.Platform.getPlatformByName(args.platform)
simulation = ufed.simulation(pdb.topology, system, integrator, platform)
simulation.set_positions(pdb.positions)
simulation.minimizeEnergy(tolerance=0.1*unit.kilojoules_per_mole)
positions = simulation.context.getState(getPositions=True).getPositions()
app.PDBFile.writeFile(pdb.topology, positions[:-1], open('minimized.pdb', 'w'))
simulation.set_velocities_to_temperature(temp, seed)
output = ufedmm.MultipleFiles(stdout, 'output.csv')
simulation.reporters += [
    ufedmm.StateDataReporter(output, 100, simulation.driving_force, step=True, speed=True),
    app.PDBReporter('output.pdb', 200)
]
simulation.step(nsteps)
