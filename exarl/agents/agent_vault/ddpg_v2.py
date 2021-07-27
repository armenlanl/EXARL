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
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras as keras
import random
import os
import pickle
from datetime import datetime
from exarl.utils.OUActionNoise import OUActionNoise
from exarl.utils.OUActionNoise import OUActionNoise2
from ._replay_buffer import ReplayBuffer, HindsightExperienceReplayMemory, PrioritedReplayBuffer
from exarl.utils.memory_type import MEMORY_TYPE
import exarl as erl

import exarl.utils.log as log
import exarl.utils.candleDriver as cd
logger = log.setup_logger(__name__, cd.run_params['log_level'])


@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


class DDPG(erl.ExaAgent):
    is_learner: bool

    def __init__(self, env, is_learner):
        # Distributed variables
        self.is_learner = False

        # Environment space and action parameters
        self.env = env
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        self.upper_bound = env.action_space.high
        self.lower_bound = env.action_space.low

        logger.info("Size of State Space:  {}".format(self.num_states))
        logger.info("Size of Action Space:  {}".format(self.num_actions))
        logger.info('Env upper bounds: {}'.format(self.upper_bound))
        logger.info('Env lower bounds: {}'.format(self.lower_bound))

        self.gamma = cd.run_params['gamma']
        self.tau = cd.run_params['tau']

        # model definitions
        self.actor_dense = cd.run_params['actor_dense']
        self.actor_dense_act = cd.run_params['actor_dense_act']
        self.actor_out_act = cd.run_params['actor_out_act']
        self.actor_optimizer = cd.run_params['actor_optimizer']
        self.critic_state_dense = cd.run_params['critic_state_dense']
        self.critic_state_dense_act = cd.run_params['critic_state_dense_act']
        self.critic_action_dense = cd.run_params['critic_action_dense']
        self.critic_action_dense_act = cd.run_params['critic_action_dense_act']
        self.critic_concat_dense = cd.run_params['critic_concat_dense']
        self.critic_concat_dense_act = cd.run_params['critic_concat_dense_act']
        self.critic_out_act = cd.run_params['critic_out_act']
        self.critic_optimizer = cd.run_params['critic_optimizer']
        self.tau = cd.run_params['tau']
        self.replay_buffer_type = cd.run_params['replay_buffer_type']

        std_dev = 0.2
        ave_bound = (self.upper_bound + self.lower_bound) / 2
        print('ave_bound: {}'.format(ave_bound))
        self.ou_noise = OUActionNoise(mean=ave_bound, std_deviation=float(std_dev) * np.ones(1))

        # Not used by agent but required by the learner class
        self.epsilon = cd.run_params['epsilon']
        self.epsilon_min = cd.run_params['epsilon_min']
        self.epsilon_decay = cd.run_params['epsilon_decay']

        # Experience data
        self.buffer_capacity = cd.run_params['buffer_capacity']
        self.batch_size = cd.run_params['batch_size']
        # self.buffer_counter = cd.run_params['buffer_counter']

        if self.replay_buffer_type == MEMORY_TYPE.UNIFORM_REPLAY:
            self.memory = ReplayBuffer(self.buffer_capacity, self.num_states, self.num_actions)
        elif self.replay_buffer_type == MEMORY_TYPE.PRIORITY_REPLAY:
            self.memory = PrioritedReplayBuffer(self.buffer_capacity, self.num_states, self.num_actions, self.batch_size)
        elif self.replay_buffer_type == MEMORY_TYPE.HINDSIGHT_REPLAY: # TODO: Double check if the environment has goal state
            self.memory = HindsightExperienceReplayMemory(self.buffer_capacity, self.num_states, self.num_actions)
        else:
            print("Unrecognized replay buffer please specify 'uniform, priority or hindsight', using default uniform sampling")
            self.memory = ReplayBuffer(self.buffer_capacity, self.num_states, self.num_actions) #TODO : maybe exit()  


        # Setup TF configuration to allow memory growth
        # tf.keras.backend.set_floatx('float64')
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(sess)

        # Training model only required by the learners
        if self.is_learner:
            self.actor_model = self.get_actor()
            self.critic_model = self.get_critic()
            self.target_actor = self.get_actor()
            self.target_critic = self.get_critic()
            self.target_actor.set_weights(self.actor_model.get_weights())
            self.target_critic.set_weights(self.critic_model.get_weights())
        else:
            with tf.device('/CPU:0'):
                self.target_actor = self.get_actor()
                self.target_critic = self.get_critic()

        # Learning rate for actor-critic models
        self.critic_lr = cd.run_params['critic_lr']
        self.actor_lr = cd.run_params['actor_lr']
        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)
        np.random.seed(0) #
        tf.random.set_seed(0)

    def remember(self, state, action, reward, next_state, done):
        # If the counter exceeds the capacity then
        self.memory.store(state, action, reward, next_state, done)

    # @tf.function
    def update_grad(self, state_batch, action_batch, reward_batch, next_state_batch,terminal_batch,b_idx=None,weights=None):
        # Training and updating Actor & Critic networks.
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch, training=True)
            y = reward_batch + self.gamma * self.target_critic(
                [next_state_batch, target_actions], training=True
            )* (1- terminal_batch)
            critic_value = self.critic_model([state_batch, action_batch], training=True)
            #critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
            critic_loss = keras.losses.MSE(y , critic_value)

            if self.replay_buffer_type == MEMORY_TYPE.PRIORITY_REPLAY:
                critic_loss = weights * critic_loss
                error = np.abs(tf.squeeze(y - critic_value).numpy())
                self.memory.batch_update(b_idx, error)
                

        logger.warning("Critic loss: {}".format(critic_loss))
        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = self.actor_model(state_batch, training=True)
            critic_value = self.critic_model([state_batch, actions], training=True)
            actor_loss = -tf.math.reduce_mean(critic_value)
            # actor_loss = tf.math.reduce_mean(critic_value)

        logger.warning("Actor loss: {}".format(actor_loss))
        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor_model.trainable_variables)
        )

    def get_actor(self):
        # State as input
        inputs = layers.Input(shape=(self.num_states,))
        # first layer takes inputs
        out = layers.Dense(self.actor_dense[0], activation=self.actor_dense_act)(inputs)
        # loop over remaining layers
        for i in range(1, len(self.actor_dense)):
            out = layers.Dense(self.actor_dense[i], activation=self.actor_dense_act)(out)
        # output layer has dimension actions, separate activation setting
        out = layers.Dense(self.num_actions, activation=self.actor_out_act,
                           kernel_initializer=tf.random_uniform_initializer())(out)
        outputs = layers.Lambda(lambda i: i * self.upper_bound)(out)
        model = tf.keras.Model(inputs, outputs)
        # model.summary()

        return model

    def get_critic(self):
        # State as input
        state_input = layers.Input(shape=self.num_states)
        # first layer takes inputs
        state_out = layers.Dense(self.critic_state_dense[0],
                                 activation=self.critic_state_dense_act)(state_input)
        # loop over remaining layers
        for i in range(1, len(self.critic_state_dense)):
            state_out = layers.Dense(self.critic_state_dense[i],
                                     activation=self.critic_state_dense_act)(state_out)

        # Action as input
        action_input = layers.Input(shape=self.num_actions)

        # first layer takes inputs
        action_out = layers.Dense(self.critic_action_dense[0],
                                  activation=self.critic_action_dense_act)(action_input)
        # loop over remaining layers
        for i in range(1, len(self.critic_action_dense)):
            action_out = layers.Dense(self.critic_action_dense[i],
                                      activation=self.critic_action_dense_act)(action_out)

        # Both are passed through seperate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out])

        # assumes at least 2 post-concat layers
        # first layer takes concat layer as input
        concat_out = layers.Dense(self.critic_concat_dense[0],
                                  activation=self.critic_concat_dense_act)(concat)
        # loop over remaining inner layers
        for i in range(1, len(self.critic_concat_dense) - 1):
            concat_out = layers.Dense(self.critic_concat_dense[i],
                                      activation=self.critic_concat_dense_act)(concat_out)

        # last layer has different activation
        concat_out = layers.Dense(self.critic_concat_dense[-1], activation=self.critic_out_act,
                                  kernel_initializer=tf.random_uniform_initializer())(concat_out)
        outputs = layers.Dense(1)(concat_out)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)
        # model.summary()

        return model

    def _convert_to_tensor(self, state_batch, action_batch, reward_batch, next_state_batch, terminal_batch):
        state_batch = tf.convert_to_tensor(state_batch,dtype=tf.float32)
        action_batch = tf.convert_to_tensor(action_batch,dtype=tf.float32)
        reward_batch = tf.convert_to_tensor(reward_batch,dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(next_state_batch,dtype=tf.float32)
        terminal_batch = tf.convert_to_tensor(terminal_batch,dtype=tf.float32)
        return state_batch, action_batch, reward_batch, next_state_batch, terminal_batch

    def generate_data(self):

        if self.replay_buffer_type == MEMORY_TYPE.UNIFORM_REPLAY:
            state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.memory.sample_buffer(self.batch_size) #done_batch might improve experience
            state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self._convert_to_tensor(state_batch, action_batch, reward_batch, next_state_batch, terminal_batch)
            yield state_batch, action_batch, reward_batch, next_state_batch, terminal_batch

        elif self.replay_buffer_type == MEMORY_TYPE.PRIORITY_REPLAY:
            state_batch, action_batch, reward_batch, next_state_batch, terminal_batch , btx_idx, weights = self.memory.sample_buffer(self.batch_size)
            state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self._convert_to_tensor(state_batch, action_batch, reward_batch, next_state_batch,terminal_batch)
            weights = tf.convert_to_tensor(weights,dtype=tf.float32)
            yield state_batch, action_batch, reward_batch, next_state_batch, terminal_batch, btx_idx, weights
        else:
            raise ValueError('Support for the replay buffer type not implemented yet!')

    def train(self, batch):
        if self.is_learner:
            logger.warning('Training...')
            if self.replay_buffer_type == MEMORY_TYPE.UNIFORM_REPLAY:
                self.update_grad(batch[0], batch[1], batch[2], batch[3], batch[4])
            elif self.replay_buffer_type == MEMORY_TYPE.PRIORITY_REPLAY:
                self.update_grad(batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6])

    def target_train(self):
        # Update the target model
        model_weights = self.actor_model.get_weights()
        target_weights = self.target_actor.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = self.tau * model_weights[i] + (1 - self.tau) * target_weights[i]
        self.target_actor.set_weights(target_weights)

        model_weights = self.critic_model.get_weights()
        target_weights = self.target_critic.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = self.tau * model_weights[i] + (1 - self.tau) * target_weights[i]
        self.target_critic.set_weights(target_weights)

    def action(self, state):
        policy_type = 1
        tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)

        sampled_actions = tf.squeeze(self.target_actor(tf_state))
        noise = self.ou_noise()
        sampled_actions_wn = sampled_actions.numpy() + noise
        legal_action = sampled_actions_wn
        isValid = self.env.action_space.contains(sampled_actions_wn)
        if isValid == False:
            legal_action = np.random.uniform(low=self.lower_bound, high=self.upper_bound, size=(self.num_actions,))
            policy_type = 0
            logger.warning('Bad action: {}; Replaced with: {}'.format(sampled_actions_wn, legal_action))
            logger.warning('Policy action: {}; noise: {}'.format(sampled_actions, noise))

        return_action = np.squeeze(legal_action)
        #print(return_action)
        #exit()
        #return_action = tf.convert_to_tensor([legal_action],)
        logger.warning('Legal action:{}'.format(return_action))

        return return_action, policy_type

    # For distributed actors #
    def get_weights(self):
        return self.target_actor.get_weights()

    def set_weights(self, weights):
        self.target_actor.set_weights(weights)

    def set_learner(self):
        self.is_learner = True
        self.actor_model = self.get_actor()
        self.critic_model = self.get_critic()
        self.target_actor = self.get_actor()
        self.target_critic = self.get_critic()
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())

    # Extra methods
    def update(self):
        print("Implement update method in ddpg.py")

    def load(self, filename):
        layers = self.target_actor.layers
        with open(filename, "rb") as f:
            pickle_list = pickle.load(f)

        for layerId in range(len(layers)):
            assert layers[layerId].name == pickle_list[layerId][0]
            layers[layerId].set_weights(pickle_list[layerId][1])

    def save(self, filename):
        layers = self.target_actor.layers
        pickle_list = []
        for layerId in range(len(layers)):
            weigths = layers[layerId].get_weights()
            pickle_list.append([layers[layerId].name, weigths])

        with open(filename, "wb") as f:
            pickle.dump(pickle_list, f, -1)

    def monitor(self):
        print("Implement monitor method in ddpg.py")

    def set_agent(self):
        print("Implement set_agent method in ddpg.py")

    def print_timers(self):
        print("Implement print_timers method in ddpg.py")

    def epsilon_adj(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
