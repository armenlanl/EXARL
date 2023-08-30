# This material was prepared as an account of work sponsored by an agency of the
# United States Government.  Neither the United States Government nor the United
# States Department of Energy, nor Battelle, nor any of their employees, nor any
# jurisdiction or organization that has cooperated in the development of these
# materials, makes any warranty, express or implied, or assumes any legal
# liability or responsibility for the accuracy, completeness, or usefulness or
# any information, apparatus, product, software, or process disclosed, or
# represents that its use would not infringe privately owned rights. Reference
# herein to any specific commercial product, process, or service by trade name,
# trademark, manufacturer, or otherwise does not necessarily constitute or imply
# its endorsement, recommendation, or favoring by the United States Government
# or any agency thereof, or Battelle Memorial Institute. The views and opinions
# of authors expressed herein do not necessarily state or reflect those of the
# United States Government or any agency thereof.
#                 PACIFIC NORTHWEST NATIONAL LABORATORY
#                            operated by
#                             BATTELLE
#                             for the
#                   UNITED STATES DEPARTMENT OF ENERGY
#                    under Contract DE-AC05-76RL01830
import gym
import time
import numpy as np
import sys
import json
import exarl as erl
from datetime import datetime
# from envs.env_vault.computePI import computePI as cp
from exarl.base.comm_base import ExaComm
from exarl.utils.introspect import introspectTrace
from exarl.utils.globals import ExaGlobals

run_name = str(ExaGlobals.lookup_params('experiment_id'))
now = datetime.now()
NAME = now.strftime("%d_%m_%Y_%H-%M-%S_")

run_name = NAME + run_name

def computePI(N, new_comm):
    h = 1.0 / N
    s = 0.0
    rank = new_comm.rank
    size = new_comm.size
    for i in range(rank, N, size):
        x = h * (i + 0.5)
        s += 4.0 / (1.0 + x**2)
    return s * h


class ExaPendulumTuple(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()
        self.env_comm = ExaComm.env_comm
        self.env = gym.make('Pendulum-v1', g=9.81)
        self.action_space = self.env.action_space
        
        self.WCT = 0
        self.reward = 0
        self.reward_cum = 0
        self.RUN_TIME  = int(ExaGlobals.lookup_params('n_steps'))
        
        # Defining tuple state space
        self.mask = gym.spaces.Dict({})
        self.end_traj = gym.spaces.Discrete(1)
        self.state = self.env.observation_space
        for i in range(100):
            self.mask[i] = gym.spaces.Discrete(65535)
        self.observation_space = gym.spaces.Tuple((self.state, self.end_traj, self.mask))

        self.upper_bound = self.env.action_space.high
        self.lower_bound = self.env.action_space.low

    @introspectTrace()
    def step(self, action):
        self.WCT+=1
        next_state, reward, done, _, info = self.env.step(action)
        time.sleep(0)  # Delay in seconds

        rank = self.env_comm.rank
        if rank == 0:
            N = 100
        else:
            N = None

        N = self.env_comm.bcast(N, 0)
        myPI = computePI(N, self.env_comm)  # Calls python function
        # myPI = cp.compute_pi(N, self.env_comm) # Calls C++ function
        PI = self.env_comm.reduce(myPI, sum, 0)

        # if self.env_comm.rank == 0:
        #     print(PI)  # Print PI for verification

        with open("./outputs/pend/" + run_name + "_" + "env", "a") as myfile:
            myfile.write(
            str(action[0]) + ' ' + 
            str(reward) + ' ' +
            str(next_state[0]) + ' ' +
            str(next_state[1]) + ' ' +
            str(next_state[2]) +
            '\n')
        # self.reward_cum += reward
        # if (self.WCT >= self.RUN_TIME or done == 1):
        #     self.reward = self.reward_cum 
        #     done = True
        self.reward = reward
        # print("Reward: ", self.reward)

        next_tuple_state = (next_state, self.end_traj, self.mask)

        return next_tuple_state, self.reward, done, info

    def reset(self):
        # self.env._max_episode_steps=self._max_episode_steps
        # print('Max steps: %s' % str(self._max_episode_steps))
        self.reward = 0
        self.reward_cum = 0
        self.WCT = 0
        default_reset = self.env.reset()
        state_tuple = (default_reset[0], self.end_traj, self.mask)
        return state_tuple, {}

    def render(self, mode='human', close=False):
        return self.env.render()

    def set_env(self):
        print('Use this function to set hyper-parameters, if any')
