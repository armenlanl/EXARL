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
# from envs.env_vault.computePI import computePI as cp
from exarl.base.comm_base import ExaComm
from exarl.utils.introspect import introspectTrace


def computePI(N, new_comm):
    h = 1.0 / N
    s = 0.0
    rank = new_comm.rank
    size = new_comm.size
    for i in range(rank, N, size):
        x = h * (i + 0.5)
        s += 4.0 / (1.0 + x**2)
    return s * h


class ExaMountainCarContinuous(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()
        self.env_comm = ExaComm.env_comm
        self.env = gym.make("MountainCarContinuous-v0")
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.upper_bound = self.env.action_space.high
        self.lower_bound = self.env.action_space.low

    @introspectTrace()
    def step(self, action):
        next_state, reward, done, _, info = self.env.step(action)
        time.sleep(0)  # Delay in seconds

        rank = self.env_comm.rank
        if rank == 0:
            N = 100
        else:
            N = None

        # print("Position: ", next_state[0])
        # print("Velocity: ", next_state[1])

        # if reward < 100:
        #     reward = -1
        # else:
        #     reward = 100

        reward_pos = next_state[0]

        # reward_pos = reward_pos/18

        reward_velo = abs(next_state[1])

        print("Reward Env: ", reward)
        reward = reward + reward_pos + reward_velo
        print("Modified Reward: ", reward)

        reward = reward_velo

        N = self.env_comm.bcast(N, 0)
        myPI = computePI(N, self.env_comm)  # Calls python function
        # myPI = cp.compute_pi(N, self.env_comm) # Calls C++ function
        PI = self.env_comm.reduce(myPI, sum, 0)

        # if self.env_comm.rank == 0:
        #     print(PI)  # Print PI for verification

        return next_state, reward, done, info

    def reset(self):
        # self.env._max_episode_steps=self._max_episode_steps
        # print('Max steps: %s' % str(self._max_episode_steps))
        
        return self.env.reset()

    def render(self, mode='human', close=False):
        return self.env.render()

    def set_env(self):
        print('Use this function to set hyper-parameters, if any')
