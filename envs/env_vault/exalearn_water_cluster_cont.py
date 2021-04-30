import exarl.mpi_settings as mpi_settings
import gym
import time
from mpi4py import MPI
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
import utils.log as log
import utils.candleDriver as cd
logger = log.setup_logger(__name__, cd.run_params['log_level'])

from ase.io import read, write
from ase import Atom, Atoms

from math import log10
import subprocess
import os
import math
import tempfile
from shutil import copyfile
import torch
import itertools
import random

from schnetpack import AtomsData
from schnetpack import AtomsLoader

from schnetpack.environment import SimpleEnvironmentProvider
from schnetpack import AtomsLoader

# python implementation of TTM
import scipy.optimize as opt
from ase.optimize.optimize import Optimizer
from ttm import TTM
from ttm.flib import ttm_from_f2py
from ase.calculators.calculator import Calculator
from ttm.ase import TTMCalculator

# Python TTM
class Converged(Exception):
    pass


class OptimizerConvergenceError(Exception):
    pass


class SciPyOptimizer(Optimizer):
    def __init__(self, atoms, logfile='-', trajectory=None,
                 callback_always=False, alpha=70.0, master=None,
                 force_consistent=None):
        restart = None
        Optimizer.__init__(self, atoms, restart, logfile, trajectory,
                           master, force_consistent=force_consistent)
        self.force_calls = 0
        self.callback_always = callback_always
        self.H0 = alpha

        self.pot_function = ttm_from_f2py
        self.model = 21

    def x0(self):
        return self.atoms.get_positions().reshape(-1)

    def f(self, x):
        self.atoms.set_positions(x.reshape(-1, 3))
        self.TTM_calc()
        return (self.energy/self.H0, self.gradients)

    def fprime(self, x):
        self.atoms.set_positions(x.reshape(-1, 3))
        self.force_calls += 1

        if self.callback_always:
            self.callback(x)

        return - self.gradients.reshape(-1,3) / self.H0

    def callback(self, x):
        self.TTM_calc()
        f = -self.gradients.reshape(-1,3)
        self.log(f)
        self.call_observers()
        if self.converged(f):
            raise Converged
        self.nsteps += 1

    def run(self, fmax=0.05, steps=100000000):
        if self.force_consistent is None:
            self.set_force_consistent()
        self.fmax = fmax
        #self.callback(None)
        try:
            self.call_fmin(fmax / self.H0, steps)
        except Converged:
            pass

    @staticmethod
    def ttm_ordering(coords):
        atom_order = []
        for i in range(0, coords.shape[0], 3):
            atom_order.append(i)
        for i in range(0, coords.shape[0], 3):
            atom_order.append(i+1)
            atom_order.append(i+2)
        return coords[atom_order,:]

    @staticmethod
    def normal_water_ordering(coords):
        atom_order = []
        Nw = int(coords.shape[0] / 3)
        for i in range(0, Nw, 1):
            atom_order.append(i)
            atom_order.append(Nw+2*i)
            atom_order.append(Nw+2*i+1)
        return coords[atom_order,:]

    def TTM_calc(self, *args):
        coords = self.ttm_ordering(self.atoms.get_positions())
        gradients, self.energy = self.pot_function(self.model, np.asarray(coords).T)
        self.gradients = self.normal_water_ordering(gradients.T).reshape(-1)


    def TTM_grad(self, *args):
        return self.gradients


    def dump(self, data):
        pass

    def load(self):
        pass

    def call_fmin(self, fmax, steps):
        raise NotImplementedError

class SciPyFminLBFGSB(SciPyOptimizer):
    """Quasi-Newton method (Broydon-Fletcher-Goldfarb-Shanno)"""
    def call_fmin(self, fmax, steps):
        output = opt.minimize(self.f, self.x0(), method='L-BFGS-B', jac=self.TTM_grad, options={'maxiter':1000, 'ftol':1e-8, 'gtol':1e-8})


# SchNet
def load_data(atoms, idx=0):
    at = atoms
    new_dataset={}
    atom_positions = at.positions.astype(np.float32)
    atom_positions -= at.get_center_of_mass() 
    environment_provider = SimpleEnvironmentProvider()
    nbh_idx, offsets = environment_provider.get_environment(at)
    new_dataset['_atomic_numbers'] = torch.LongTensor(at.numbers.astype(np.int))
    new_dataset['_positions'] = torch.FloatTensor(atom_positions)
    new_dataset['_cell'] = torch.FloatTensor(at.cell)#.astype(np.float32))
    new_dataset['_neighbors'] = torch.LongTensor(nbh_idx.astype(np.int))
    new_dataset['_cell_offset'] = torch.FloatTensor(offsets.astype(np.float32))
    new_dataset['_idx'] = torch.LongTensor(np.array([idx], dtype=np.int))
    return AtomsLoader([new_dataset], batch_size=1)


def get_activation(name, activation={}):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def get_state_embedding(model,structure):
    data_loader =  load_data(structure, idx=0)
    activation = {}
    logger.debug('torch.no_grad()')
    with torch.no_grad():
        for batch in data_loader:
            model.module.output_modules[0].standardize.register_forward_hook(get_activation('standardize',activation))
            output = model(batch)
        state_embedding = activation['standardize'].cpu().detach().numpy()
        output = output['energy'].cpu().detach().numpy()

    state_embedding = state_embedding.flatten()
    state_order = np.argsort(state_embedding[::3])
    state_embedding = np.sort(state_embedding)
    state_embedding = np.insert(state_embedding, 0, float(np.sum(state_embedding)), axis=0)

    return state_embedding, state_order, output[0][0]


def update_cluster(structure, action=[0,0,0]):
    temp_struct=structure.copy()
    natoms = len(structure)

    # Actions
    cluster_id = math.floor(action[0])
    rotation_z = float(action[1])
    translation_action = float(action[2])  # applied equally to (x,y,z)

    # Create water cluster from action
    O_idx = cluster_id *3
    atoms = Atoms()
    for i in range(3):
        atoms.append(temp_struct[O_idx + i])

    # rotation
    O_coords = structure[O_idx].position
    H_coords = [structure[O_idx+1].position, structure[O_idx+2].position]
    # Calculate bisector vector along two O--H bonds.
    u = np.array(H_coords[0]) - np.array(O_coords)
    v = np.array(H_coords[1]) - np.array(O_coords)
    bisector_vector = (np.linalg.norm(v) * u) + (np.linalg.norm(u) * v)

    # Apply rotation through the z-axis of the water molecule.
    atoms.rotate(rotation_z, v=bisector_vector, center=O_coords)
    atoms.translate(translation_action)

    # Update structure
    for i in range(3):
        structure[O_idx + i].position = atoms[i].position

    return structure

def write_structure(PATH, structure, energy=0.0):
    nice_struct = [str(n).replace('[','').replace(']','') for n in structure.positions]
    nice_struct = ['  '.join(x) for x in list(zip(structure.get_chemical_symbols(),nice_struct))]
    pos='\n'.join(nice_struct)
    with open(PATH, 'w') as f:
        f.writelines(str(len(structure))+'\n')
        f.writelines(str(energy)+'\n')
        f.writelines(pos)

def write_csv(path, rank, data):
    if rank > 0:
        data = [str(x) for x in data]
        data = ','.join(data)
        with open(os.path.join(path,'rank{}.csv'.format(rank)), 'a') as f:
            f.writelines(data+'\n')


def get_dataset_info(xyz):
    with open(xyz, "r") as f:
        n_atoms = f.readline()       #atoms per cluster
        n_lines = 2 + int(n_atoms)   #lines per cluster (#atoms + energy + coords)
    with open(xyz, "r") as f:
        lines = f.readlines()
    energies = np.array(lines[1::n_lines], dtype=np.float32)
    return int(n_atoms), energies


class WaterCluster(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):#, cfg='envs/env_vault/env_cfg/env_setup.json'):
        super().__init__()

        # Default
        natoms = 3
        self.init_structure = None
        self.current_structure = None
        self.nclusters = 0.0
        self.inital_state = 0.0
        self.current_state = 0.0
        #self.reward_scale = 2.0
        self.initial_energy = 0 
        self.current_energy = 0
        self.episode = 0
        self.steps = 0
        self.lowest_energy = 0
        self.streak = 0
        self.max_streak = 1 
        self.nstructures = 0
        self.energies = []
        self.episode_energies = []

        #############################################################
        # Setup water molecule application (should be configurable)
        #############################################################
        self.app_dir = cd.run_params['app_dir']
        logger.debug('Using app_dir: {}'.format(self.app_dir))
        self.app_name = 'main.x'
        self.app = os.path.join(self.app_dir, self.app_name)
        self.env_input_name = cd.run_params['env_input_name']
        self.env_input_dir = cd.run_params['env_input_dir']
        self.env_input = os.path.join(self.env_input_dir, self.env_input_name)
        self.nclusters, self.energies = get_dataset_info(self.env_input)
        self.nstructures = len(self.energies)
        logger.warning('Number of structures in database: {}'.format(self.nstructures))
        logger.warning('Initial energy: {}'.format(self.energies[0]))
        self.output_dir = cd.run_params['output_dir']
        self.calc = TTMCalculator()

        # Schnet encodering model
        self.schnet_model_pfn = cd.run_params['schnet_model_pfn']
        model = torch.load(self.schnet_model_pfn, map_location='cpu')
        self.schnet_model =  torch.nn.DataParallel(model.module)

        # Read initial XYZ file
        (self.current_ase, self.nclusters) = self._load_structure(self.env_input, 0)
        self.current_ase.calc = self.calc
        self.inital_state, self.state_order, _ = get_state_embedding(self.schnet_model, self.current_ase) 
        self.initial_energy = self.energies[0]
        self.current_energy = self.initial_energy
        self.lowest_energy = self.initial_energy

        # State of the current setup
        self.init_structure = self.env_input
        self.current_structure = self.init_structure

        # Env state output: based on the SchetPack molecule energy
        self.embedded_state_size = (self.nclusters+1)*natoms+1
        self.observation_space = spaces.Box(low=-500*np.ones(self.embedded_state_size),
                                            high=np.zeros(self.embedded_state_size), dtype=np.float64)

        # Actions
        self.rot_min = 80
        self.rot_max = 120
        self.rot_mean = (self.rot_max + self.rot_min)/2
        rot_width = self.rot_max - self.rot_min
        self.trans_min = 0.5
        self.trans_max = 0.7
        self.trans_mean = (self.trans_max + self.trans_min)/2
        trans_width = self.trans_max - self.trans_min
        self.action_space = spaces.Box(low=np.array([-(self.nclusters)/2, -rot_width/2, -trans_width/2]),
                                       high=np.array([(self.nclusters)/2, +rot_width/2, +trans_width/2]), dtype=np.float64)

    def _load_structure(self, env_input, n):
        # Read initial XYZ file
        logger.debug('Env Input: {}'.format(env_input))
        # static read
        #structure = read(env_input, parallel=False)
        # random read
        logger.debug('Random structure index: {}'.format(n))
        structure = read(env_input, index=f'{n}:{n+1}', parallel=False)[0]
        logger.debug('Structure: {}'.format(structure))
        nclusters = (''.join(structure.get_chemical_symbols()).count("OHH")) - 1
        logger.debug('Number of atoms: %s' % len(structure))
        logger.debug('Number of water clusters: %s ' % (nclusters + 1))
        return (structure, nclusters)

    def step(self, action):
        logger.debug('Env::step()')
        self.steps += 1
        logger.debug('Env::step(); steps[{0:3d}]'.format(self.steps))
        logger.debug('Current energy:{}'.format(self.current_energy))
        logger.debug('Current energy:{}'.format(self.current_energy))
        action = action[0]
        logger.debug('Action:{}'.format(action))
        
        # Initialize outut
        done = False
        #energy = np.random.normal(self.current_energy, 0.01)  # Default energy
        reward = round(np.random.normal(0.0, 0.05),4)  # Default penalty
        bad_reward = round(np.random.normal(-5.0, 0.1),4) # Negative penalty

        # Extract actions
        natoms = 3
        cluster_id = math.floor(action[0]+self.nclusters/2+0.5)
        cluster_id = self.state_order[cluster_id]
        rotation_z = round(float(action[1])+self.rot_mean,2)
        translation = round(float(action[2]+self.trans_mean ),4)  # (x,y,z)
        actions = [cluster_id, rotation_z, translation]
        
        # read in structure as ase atom object
        try:
            self.current_ase = read(self.current_structure, parallel=False)
        except:
            logger.warning("couldn't read structure as ase atoms object")
            done = True
            self.current_energy = 0   
            self.current_state = np.zeros(self.embedded_state_size)
            write_csv(self.output_dir, mpi_settings.agent_comm.rank, [self.nclusters+1, mpi_settings.agent_comm.rank, self.episode, self.steps, 0, 0, 1, 0, 0, reward, done])
            return self.current_state, reward, done, {}

        # take actions
        self.current_ase = update_cluster(self.current_ase, actions)

        # Save structure in xyz format
        self.current_structure = os.path.join(self.output_dir,'rotationz_rank{}_episode{}_steps{}.xyz'.format(mpi_settings.agent_comm.rank, self.episode, self.steps))
        write_structure(self.current_structure, self.current_ase)
        #fix_structure_file(self.current_structure)

        # Run the process
        min_xyz = os.path.join(self.output_dir,'minimum_rank{}.xyz'.format(mpi_settings.agent_comm.rank))
        env_out = subprocess.Popen([self.app, self.current_structure, min_xyz], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout, stderr = env_out.communicate()
        stdout = stdout.decode('latin-1').splitlines()

        # Check for clear problems
        if any("Error in the det" in s for s in stdout):
            logger.warning("\tEnv::step(); !!! Error in the det !!!")
            os.remove(self.current_structure)
            done = True
            self.current_state = np.zeros(self.embedded_state_size)
            reward = bad_reward
            self.current_energy = 0 
            write_csv(self.output_dir, mpi_settings.agent_comm.rank, [self.nclusters+1, mpi_settings.agent_comm.rank, self.episode, self.steps, cluster_id, rotation_z, translation, 0, 0, reward, done])
            return self.current_state, reward, done, {}
        '''

        try:
            #self.current_ase.calc = self.calc
            dyn = SciPyFminLBFGSB(self.current_ase)
            dyn.run(fmax=1e-2)
            energy = self.calc.get_potential_energy(self.current_ase)
            logger.debug('energy from ttm {}'.format(energy))
        except:
            #os.remove(self.current_structure)
            done = True
            self.current_state = np.zeros(self.embedded_state_size)
            write_csv(self.output_dir, mpi_settings.agent_comm.rank, [self.nclusters, mpi_settings.agent_comm.rank, self.episode, self.steps, cluster_id, rotation_z, translation, self.current_energy, self.current_state[0], reward, done])
            return self.current_state, reward, done, {}
        '''

        # Reward is currently based on the potential energy
        #lowest_energy_xyz = ''
        self.current_ase = read(min_xyz, parallel=False)
        energy = round(float(stdout[-1].split()[-1]),4)
        self.current_state, self.state_order, schnet_energy = get_state_embedding(self.schnet_model, self.current_ase)
        schnet_energy = round(schnet_energy, 4)

        logger.debug('lowest_energy:{}'.format(self.lowest_energy))
        logger.debug('energy:{}'.format(energy))

        if  round(self.lowest_energy,4)>round(energy,4):
            self.lowest_energy=energy
        #    done = True
        #    reward = (self.current_energy - energy)*self.episodes
            #lowest_energy_xyz = os.path.join(self.output_dir,'rotationz_rank{}_episode{}_steps{}_energy{}.xyz'.format(
            #    mpi_settings.agent_comm.rank, self.episode, self.steps,round(self.lowest_energy,4)))
            logger.info("\t Found lower energy:{}".format(energy))
            #write_structure(lowest_energy_xyz, self.current_ase, self.current_energy)
        #    return self.current_state, reward, done, {}

        # End episode if the structure is unstable
        if energy > max(self.energies) * 0.95:
            logger.warning('energy too high {}'.format(energy))
            done = True
            reward = bad_reward
            #self.current_state = np.zeros(self.embedded_state_size)
            write_csv(self.output_dir, mpi_settings.agent_comm.rank, [self.nclusters+1, mpi_settings.agent_comm.rank, self.episode, self.steps, cluster_id, rotation_z, translation, energy, schnet_energy, reward, done])
            return self.current_state, reward, done, {}

        logger.debug('Pre-step current energy:{}'.format(self.current_energy))
        logger.debug('Energy:{}'.format(energy))
        logger.debug('Reward:{}'.format(reward))

        # Check with Schnet predictions
        #schnet_energy = self.current_state[0]
        #energy_mape = np.abs(energy-schnet_energy)/(energy+schnet_energy)
        #if energy_mape>0.01:
        #    logger.debug('Large difference model predict and Schnet MAPE :{}'.format(energy_mape))

        # End episode if the same structure is found max_streak times in a row
        if round(self.current_energy,3) == round(energy,3):
            # Return values
            self.streak += 1 
            if self.streak == self.max_streak:
                logger.warning('episode {} step {}: max streak reached {}'.format(self.episode, self.steps, self.max_streak))
                done = True
                #self.current_state = np.zeros(self.embedded_state_size)
                reward = bad_reward
                write_csv(self.output_dir, mpi_settings.agent_comm.rank, [self.nclusters+1, mpi_settings.agent_comm.rank, self.episode, self.steps, cluster_id, rotation_z, translation, energy, schnet_energy, reward, done])
                return self.current_state, reward, done, {}
            
        else:
            self.streak = 0
            # only give reward if a different structure is found, else 0ish
            #reward = self.current_energy - energy

        # Set reward to normalized SchNet energy (first value in state) 
        reward = (self.current_energy - energy)#*(1+self.steps/100)  #/ self.initial_energy 
        reward = round(reward,4)
        # Update current energy    
        self.current_energy = energy       
        self.episode_energies += [self.current_energy]

        write_structure(self.current_structure, self.current_ase, self.current_energy)


        write_csv(self.output_dir, mpi_settings.agent_comm.rank, [self.nclusters+1, mpi_settings.agent_comm.rank, self.episode, self.steps, cluster_id, rotation_z, translation, self.current_energy, schnet_energy, reward, done])

        logger.debug('Reward:{}'.format(reward))
        logger.debug('Energy:{}'.format(energy))
        return self.current_state, reward, done, {}

    def reset(self):
        logger.info("Resetting the environemnts.")
        logger.info("Current lowest energy: {}".format(self.lowest_energy))

        # delete all files from last episode of the rank (not lowest energy files)
        files_in_directory = os.listdir(self.output_dir)
        filtered_files = [file for file in files_in_directory if (file.endswith(".xyz")) and ('rank{}_'.format(mpi_settings.agent_comm.rank) in file) and ('-' not in file)]
        for file in filtered_files:
            path_to_file = os.path.join(self.output_dir, file)
            os.remove(path_to_file)

        self.episode += 1
        self.steps = 0
        self.streak = 0
        logger.debug('Env::reset(); episode[{0:4d}]'.format(self.episode, self.steps))
        n = random.randint(0, self.nstructures)
        logger.warning('Restart episode {}: start from {} with energy {}'.format(self.episode, n, self.energies[n]))
        (self.current_ase, self.nclusters) = self._load_structure(self.env_input, n=n)
        self.current_ase.calc = self.calc
        #self.current_structure = self.init_structure
        self.current_structure = os.path.join(self.output_dir,'rotationz_rank{}_episode{}_steps{}.xyz'.format(mpi_settings.agent_comm.rank, self.episode, self.steps))

        self.current_state, self.state_order, schnet_energy = get_state_embedding(self.schnet_model, self.current_ase)
        schnet_energy = round(schnet_energy, 4)
        self.initial_energy = round(self.energies[n], 4) 
        self.current_energy = self.initial_energy
        self.episode_energies = [self.current_energy]

        write_structure(self.current_structure, self.current_ase)
        write_csv(self.output_dir, mpi_settings.agent_comm.rank, [self.nclusters+1, mpi_settings.agent_comm.rank, self.episode, self.steps, 0, 0, 0, self.current_energy, schnet_energy, 0, False])
        logger.debug('self.current_state shape:{}'.format(self.current_state.shape))
        return self.current_state

    def render(self, mode='human'):
        return 0

    def close(self):
        return 0
