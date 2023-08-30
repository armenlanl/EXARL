# Copyright (c) 2020, Jefferson Science Associates, LLC. All Rights Reserved. Redistribution
# and use in source and binary forms, with or without modification, are permitted as a
# licensed user provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this
#    list of conditions and the following disclaimer in the documentation and/or other
#    materials provided with the distribution.
# 3. The name of the author may not be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# This material resulted from work developed under a United States Government Contract.
# The Government retains a paid-up, nonexclusive, irrevocable worldwide license in such
# copyrighted data to reproduce, distribute copies to the public, prepare derivative works,
# perform publicly and display publicly and to permit others to do so.
#
# THIS SOFTWARE IS PROVIDED BY JEFFERSON SCIENCE ASSOCIATES LLC "AS IS" AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL
# JEFFERSON SCIENCE ASSOCIATES, LLC OR THE U.S. GOVERNMENT BE LIABLE TO LICENSEE OR ANY
# THIRD PARTES FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
import numpy as np
import tensorflow as tf
import os
from mpi4py import MPI
from datetime import datetime
# import tensorflow_probability as tfp
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers

import exarl
from exarl.utils.globals import ExaGlobals
from exarl.utils.OUActionNoise import OUActionNoise
from exarl.agents.agent_vault._replay_buffer import ReplayBuffer
from exarl.agents.agent_vault._replay_buffer import nStepBuffer
logger = ExaGlobals.setup_logger(__name__)

run_name = str(ExaGlobals.lookup_params('experiment_id'))
now = datetime.now()
NAME = now.strftime("%d_%m_%Y_%H-%M-%S_")

run_name = NAME + run_name

def softmax(x):
    return(np.exp(x)/np.exp(x).sum())

class KerasGraphTD3RLConst(exarl.ExaAgent):

    def __init__(self, env, is_learner, **kwargs):
        """ Define all key variables required for all agent """

        self.is_learner = is_learner
        # Get env info
        super().__init__(**kwargs)
        self.env = env
        self.num_states = env.observation_space.shape#[0]
        self.adj_shape = env.observation_space[0].shape
        self.adj_shape = (self.adj_shape[0], int(self.adj_shape[1]/2))
        self.num_actions = env.action_space.shape[0]
        self.node_count = env.metadata["node_count"]
        self.upper_bound = env.action_space.high
        self.lower_bound = env.action_space.low

        # Used to update target networks
        self.tau = ExaGlobals.lookup_params('tau')
        self.gamma = ExaGlobals.lookup_params('gamma')

        # Buffer
        self.buffer_counter = 0
        self.buffer_capacity = ExaGlobals.lookup_params('buffer_capacity')
        self.batch_size = ExaGlobals.lookup_params('batch_size')
        self.horizon = 1
        if self.horizon == 1:
            self.memory = ReplayBuffer(self.buffer_capacity, self.num_states, self.num_actions)
        else:
            self.memory = nStepBuffer(self.buffer_capacity, self.num_states, self.num_actions, self.horizon, self.gamma)
        self.per_buffer = np.ones((self.buffer_capacity, 1))

        # Model Definitions 
        self.actor_dense = ExaGlobals.lookup_params('actor_dense')
        self.actor_dense_act = ExaGlobals.lookup_params('actor_dense_act')
        self.actor_out_act = ExaGlobals.lookup_params('actor_out_act')
        self.actor_optimizer = ExaGlobals.lookup_params('actor_optimizer')
        self.critic_state_dense = ExaGlobals.lookup_params('critic_state_dense')
        self.critic_state_dense_act = ExaGlobals.lookup_params('critic_state_dense_act')
        self.critic_action_dense = ExaGlobals.lookup_params('critic_action_dense')
        self.critic_action_dense_act = ExaGlobals.lookup_params('critic_action_dense_act')
        self.critic_concat_dense = ExaGlobals.lookup_params('critic_concat_dense')
        self.critic_concat_dense_act = ExaGlobals.lookup_params('critic_concat_dense_act')
        self.critic_out_act = ExaGlobals.lookup_params('critic_out_act')
        self.critic_optimizer = ExaGlobals.lookup_params('critic_optimizer')

        # Setup Optimizers
        critic_lr = ExaGlobals.lookup_params('critic_lr')
        actor_lr = ExaGlobals.lookup_params('actor_lr')
        self.critic_optimizer1 = Adam(critic_lr, epsilon=1e-08)
        self.critic_optimizer2 = Adam(critic_lr, epsilon=1e-08)
        self.actor_optimizer = Adam(actor_lr, epsilon=1e-08)

        # Setup models
        self.actor_model = self.get_actor()
        self.target_actor = self.get_actor()
        self.target_actor.set_weights(self.actor_model.get_weights())

        tf.random.set_seed(1)
        self.critic_model1 = self.get_critic()
        self.target_critic1 = self.get_critic()
        self.target_critic1.set_weights(self.critic_model1.get_weights())

        tf.random.set_seed(1234)
        self.critic_model2 = self.get_critic()
        self.target_critic2 = self.get_critic()
        self.target_critic2.set_weights(self.critic_model2.get_weights())

        # update counting
        self.ntrain_calls = 0
        self.actor_update_freq = int(ExaGlobals.lookup_params('n_steps')/50)
        self.target_critic_update_freq = int(ExaGlobals.lookup_params('n_steps')/50)
        self.critic_update_freq = int(ExaGlobals.lookup_params('n_steps')/100)

        # Ornstein-Uhlenbeck process
        std_dev = 0.2
        ave_bound = np.zeros(1)
        self.ou_noise = OUActionNoise(mean=ave_bound, std_deviation=float(std_dev) * np.ones(1))

        # Not used by agent but required by the learner class
        self.epsilon = ExaGlobals.lookup_params('epsilon')
        self.epsilon_min = ExaGlobals.lookup_params('epsilon_min')
        self.epsilon_decay = ExaGlobals.lookup_params('epsilon_decay')
        
        self.noise = ExaGlobals.lookup_params('noise')
        self.noise_min = ExaGlobals.lookup_params('noise_min')
        self.noise_decay = ExaGlobals.lookup_params('noise_decay')
        print("Epsilon: ", self.epsilon, " min: ", self.epsilon_min, " decay: ", self.epsilon_decay)

        logger().info("TD3 buffer capacity {}".format(self.buffer_capacity))
        logger().info("TD3 batch size {}".format(self.batch_size))
        logger().info("TD3 tau {}".format(self.tau))
        logger().info("TD3 gamma {}".format(self.gamma))
        logger().info("TD3 critic_lr {}".format(critic_lr))
        logger().info("TD3 actor_lr {}".format(actor_lr))

        # Actor logs
        self.debug_mode = True
        if self.debug_mode:
            tf.config.run_functions_eagerly(True)
            self.actor_loss_log = 0.0
            self.critic1_loss_log = 0.0
            self.critic2_loss_log = 0.0
            self.actions_log = []

    @tf.function
    def train_critic(self, states, actions, rewards, next_states, dones, masks, next_masks):
        
        next_adj_mat, next_dat_mat = tf.split(next_states, num_or_size_splits=2, axis=2)
        next_actions = self.target_actor([next_adj_mat, next_dat_mat], training=False)
        
        # Add a little noise
        # noise = self.ou_noise()
        # next_actions = next_actions + noise
        noise = np.random.normal(0, self.noise, self.num_actions)
        noise = np.clip(noise, -0.5, 0.5)
        next_actions = next_actions * (1 + noise)
        self.noise = max((self.noise*self.noise_decay), self.noise_min)
       
        #Masking the invalid actions
        next_actions = tf.where(next_masks, next_actions, tf.constant(-np.inf, shape=next_actions.shape))
        next_actions = tf.nn.softmax(next_actions)
        
        new_q1 = self.target_critic1([next_adj_mat, next_dat_mat, next_actions], training=False)
        new_q2 = self.target_critic2([next_adj_mat, next_dat_mat, next_actions], training=False)
        new_q = tf.math.minimum(new_q1, new_q2)

        # Bellman equation for the q value
        q_targets = rewards + (1.0 - dones) * self.gamma * new_q

        actions = tf.where(masks, actions, tf.constant(-np.inf, shape=actions.shape))
        actions = tf.nn.softmax(actions)

        # Critic 1
        with tf.GradientTape() as tape:
            adj_mat, dat_mat = tf.split(states, num_or_size_splits=2, axis=2)
            q_values1 = self.critic_model1([adj_mat, dat_mat, actions], training=True)
            td_errors1 = q_values1 - q_targets
            critic_loss1 = tf.reduce_mean(tf.math.square(td_errors1))
            if self.debug_mode:
                self.critic1_loss_log = critic_loss1.numpy()
        gradient1 = tape.gradient(critic_loss1, self.critic_model1.trainable_variables)
        self.critic_optimizer1.apply_gradients(zip(gradient1, self.critic_model1.trainable_variables))

        # Critic 2
        with tf.GradientTape() as tape:
            adj_mat, dat_mat = tf.split(states, num_or_size_splits=2, axis=2)
            q_values2 = self.critic_model2([adj_mat, dat_mat, actions], training=True)
            td_errors2 = q_values2 - q_targets
            critic_loss2 = tf.reduce_mean(tf.math.square(td_errors2))
            if self.debug_mode:
                self.critic2_loss_log = critic_loss2.numpy()
        gradient2 = tape.gradient(critic_loss2, self.critic_model2.trainable_variables)
        self.critic_optimizer2.apply_gradients(zip(gradient2, self.critic_model2.trainable_variables))

    @tf.function
    def train_actor(self, states, masks):
        # Use Critic 1
        with tf.GradientTape() as tape:
            adj_mat, dat_mat = tf.split(states, num_or_size_splits=2, axis=2)

            actions = self.actor_model([adj_mat, dat_mat], training=True)

            # Invalid action masking with tensors
            actions = tf.where(masks, actions, tf.constant(-np.inf, shape=actions.shape))
            actions = tf.nn.softmax(actions)
            # tf.print("Actions in training: ", actions)
            q_value = self.critic_model1([adj_mat, dat_mat, actions], training=True)
            # tf.print("Q value: ", q_value)
            loss = -tf.math.reduce_mean(q_value)
            if self.debug_mode:
                self.actor_loss_log = loss.numpy()
            # tf.print("Loss: ", loss)

        gradient = tape.gradient(loss, self.actor_model.trainable_variables)
        
        # for var, g in zip(self.actor_model.trainable_variables, gradient):
        #     # in this loop g is the gradient of each layer
        #     print(f'{var.name}, shape: {g.shape}')
        #     print("gradients..")
        #     print(g)
        #     if tf.math.reduce_any(tf.math.is_nan(g)):
        #         print("Found NaN gradient here")

        self.actor_optimizer.apply_gradients(zip(gradient, self.actor_model.trainable_variables))

    def get_critic(self):
        # State as input
        adj_inputs = tf.keras.layers.Input(shape=self.adj_shape + (1,))
        # state_out = tf.keras.layers.Conv2D(32, kernel_size=5, activation="relu")(adj_inputs)
        # state_out = tf.keras.layers.Conv2D(16, kernel_size=5, activation="relu")(state_out)
        # state_out = tf.keras.layers.Conv2D(4, kernel_size=5, activation="relu")(state_out)
        state_out = tf.keras.layers.Flatten()(adj_inputs)
        state_out = tf.keras.layers.Dense(512, activation="relu")(state_out)
        state_out = tf.keras.layers.Dense(256, activation="relu")(state_out)
        state_out = tf.keras.layers.Dense(128, activation="relu")(state_out)

        dat_inputs = tf.keras.layers.Input(shape=self.adj_shape + (1,))
        # dat_out = tf.keras.layers.Conv2D(32, kernel_size=5, activation="relu")(dat_inputs)
        # dat_out = tf.keras.layers.Conv2D(16, kernel_size=5, activation="relu")(dat_out)
        # dat_out = tf.keras.layers.Conv2D(4, kernel_size=5, activation="relu")(dat_out)
        dat_out = tf.keras.layers.Flatten()(dat_inputs)
        dat_out = tf.keras.layers.Dense(512, activation="relu")(dat_out)
        dat_out = tf.keras.layers.Dense(256, activation="relu")(dat_out)
        dat_out = tf.keras.layers.Dense(128, activation="relu")(dat_out)


        state_out = tf.keras.layers.Concatenate()([state_out,dat_out])

        # first layer takes inputs
        state_out = layers.Dense(self.critic_state_dense[0],
                                    activation=self.critic_state_dense_act)(state_out)
        # loop over remaining layers
        for i in range(1, len(self.critic_state_dense)):
            state_out = layers.Dense(self.critic_state_dense[i],
                                        activation=self.critic_state_dense_act)(state_out)

        # Action as input
        # action_input = tf.keras.layers.Input(shape=(self.num_actions))
        action_input = tf.keras.layers.Input(shape=(self.num_actions,), name='action_input')

        # first layer takes inputs
        action_out = layers.Dense(self.critic_action_dense[0],
                                    activation=self.critic_action_dense_act)(action_input)
        # loop over remaining layers
        for i in range(1, len(self.critic_action_dense)):
            action_out = layers.Dense(self.critic_action_dense[i],
                                        activation=self.critic_action_dense_act)(action_out)

        # Both are passed through separate layer before concatenating
        concat = tf.keras.layers.Concatenate()([state_out, action_out])

        concat_out = layers.Dense(self.critic_concat_dense[0],
                                    activation=self.critic_concat_dense_act)(concat)
        # loop over remaining inner layers
        for i in range(1, len(self.critic_concat_dense) - 1):
            concat_out = layers.Dense(self.critic_concat_dense[i],
                                        activation=self.critic_concat_dense_act)(concat_out)

        # last layer has different activation
        concat_out = layers.Dense(self.critic_concat_dense[-1], activation=self.critic_out_act,
                                    kernel_initializer=tf.random_uniform_initializer())(concat_out)

        outputs = tf.keras.layers.Dense(1)(concat_out)

        # Outputs single value for give state-action
        model = tf.keras.Model([adj_inputs, dat_inputs, action_input], outputs)
        model.summary()

        return model

    def get_actor(self):
        """Define actor network

        Returns:
            model: actor model
        """
        # State as input

        adj_inputs = tf.keras.layers.Input(shape=self.adj_shape + (1,))
        # state_out = tf.keras.layers.Conv2D(32, kernel_size=5, activation="relu")(adj_inputs)
        # state_out = tf.keras.layers.Conv2D(16, kernel_size=5, activation="relu")(state_out)
        # state_out = tf.keras.layers.Conv2D(4, kernel_size=5, activation="relu")(state_out)
        state_out = tf.keras.layers.Flatten()(adj_inputs)
        state_out = tf.keras.layers.Dense(512, activation="relu")(state_out)
        state_out = tf.keras.layers.Dense(256, activation="relu")(state_out)
        state_out = tf.keras.layers.Dense(128, activation="relu")(state_out)

        dat_inputs = tf.keras.layers.Input(shape=self.adj_shape + (1,))
        # dat_out = tf.keras.layers.Conv2D(32, kernel_size=5, activation="relu")(dat_inputs)
        # dat_out = tf.keras.layers.Conv2D(16, kernel_size=5, activation="relu")(dat_out)
        # dat_out = tf.keras.layers.Conv2D(4, kernel_size=5, activation="relu")(dat_out)
        dat_out = tf.keras.layers.Flatten()(dat_inputs)
        dat_out = tf.keras.layers.Dense(512, activation="relu")(dat_out)
        dat_out = tf.keras.layers.Dense(256, activation="relu")(dat_out)
        dat_out = tf.keras.layers.Dense(128, activation="relu")(dat_out)

        concat = tf.keras.layers.Concatenate()([state_out, dat_out])

        # first layer takes inputs
        out = layers.Dense(self.actor_dense[0], activation=self.actor_dense_act)(concat)
        # loop over remaining layers
        for i in range(1, len(self.actor_dense)):
            out = layers.Dense(self.actor_dense[i], activation=self.actor_dense_act)(out)
        # output layer has dimension actions, separate activation setting
        out = layers.Dense(self.num_actions, activation=self.actor_out_act,
                            kernel_initializer=tf.random_uniform_initializer())(out)
        out = tf.keras.layers.BatchNormalization()(out)

        # Scale to the env problem
        outputs = tf.keras.layers.Lambda(
            lambda x: ((x + 1.0) * (self.upper_bound - self.lower_bound)) / 2.0 + self.lower_bound)(out)

        model = tf.keras.Model([adj_inputs, dat_inputs], outputs)
        model.summary()
        return model

    @tf.function
    def soft_update(self, target_weights, weights):
        for (target_weight, weight) in zip(target_weights, weights):
            target_weight.assign(weight * self.tau + target_weight * (1.0 - self.tau))

    def update(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch, masks, next_masks):
        if self.ntrain_calls % self.actor_update_freq == 0:
            self.train_actor(state_batch, masks)
        if self.ntrain_calls % self.critic_update_freq == 0:
            self.train_critic(state_batch, action_batch, reward_batch, next_state_batch, done_batch, masks, next_masks)
    
    def logging(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if not os.path.exists("./outputs/Exaalt/" + run_name + "/"):
            os.makedirs("./outputs/Exaalt/" + run_name + "/")

        with open("./outputs/Exaalt/" + run_name + "/" + run_name + "_" + "agent_" + str(rank), "a") as myfile:
            myfile.write(
            str(self.actor_loss_log) + ' ' + 
            str(self.critic1_loss_log) + ' ' +
            str(self.critic2_loss_log) +
            '\n') 
     
    def _convert_to_tensor(self, state_batch, action_batch, reward_batch, next_state_batch, terminal_batch):
        masks = []
        next_masks = []

        tf_state_batch = np.zeros(shape=(self.batch_size,self.adj_shape[0],self.adj_shape[1]*2))
        tf_next_state_batch = np.zeros(shape=(self.batch_size,self.adj_shape[0],self.adj_shape[1]*2))

        for i in range(self.batch_size):

            known_keys = [x for x in state_batch[i,2].keys() if state_batch[i,2][x] != None]
            known_keys_next = [x for x in next_state_batch[i,2].keys() if next_state_batch[i,2][x] != None]

            # only need to make mask if we do not know graph size number of keys
            mask = np.full(self.num_actions, True)
            next_mask = np.full(self.num_actions, True)

            if (len(known_keys) < self.num_actions):
                mask_indices = np.arange(len(known_keys),self.num_actions)
                mask[mask_indices] = False

            if (len(known_keys_next) < self.num_actions):
                next_mask_indices = np.arange(len(known_keys_next),self.num_actions)
                next_mask[next_mask_indices] = False

            masks.append(mask)
            next_masks.append(next_mask)

            tf_state_batch[i] = tf.convert_to_tensor(state_batch[i,0], dtype=tf.float32)
            tf_next_state_batch[i] = tf.convert_to_tensor(next_state_batch[i,0], dtype=tf.float32)
        
        action_batch = tf.convert_to_tensor(action_batch, dtype=tf.float32)
        reward_batch = tf.convert_to_tensor(reward_batch, dtype=tf.float32)
        terminal_batch = tf.convert_to_tensor(terminal_batch, dtype=tf.float32)
        return tf_state_batch, action_batch, reward_batch, tf_next_state_batch, terminal_batch, masks, next_masks

    def generate_data(self):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, masks_batch, next_masks_batch = \
            self._convert_to_tensor(*self.memory.sample_buffer(self.batch_size))
        yield state_batch, action_batch, reward_batch, next_state_batch, done_batch, masks_batch, next_masks_batch

    def train(self, batch):
        """ Method used to train """
        self.ntrain_calls += 1
        self.update(batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6])
        if self.debug_mode and self.ntrain_calls % self.critic_update_freq == 0:
            self.logging()

    def update_target(self):
        if self.ntrain_calls % self.actor_update_freq == 0:
            # print("Training Target Actor")
            self.soft_update(self.target_actor.variables, self.actor_model.variables)
        if self.ntrain_calls % self.target_critic_update_freq == 0:
            # print("Training Target Critics")
            self.soft_update(self.target_critic1.variables, self.critic_model1.variables)
            self.soft_update(self.target_critic2.variables, self.critic_model2.variables)

    def action(self, state):
        """ Method used to provide the next action using the target model """
        # print("State: ", state)
        policy_type = 1
        taskList = []
        taskList.append(state[1])
        known_keys = [x for x in state[2].keys() if state[2][x] != None]

        # weights = self.actor_model.get_weights()

        # for weight in weights:   
        #     if np.any(np.isnan(weight)):     
        #         print('Found nan weight at index {}'.format(weight.shape[0]))
        
        # create the tensor from the adj matrix and get the action size number of nodes in graph
        tf_state = tf.expand_dims(tf.convert_to_tensor(state[0]), 0)
        adj_mat, dat_mat = tf.split(tf_state, num_or_size_splits=2, axis=2)
        
        raw_output = self.actor_model([adj_mat,dat_mat])
        sampled_actions = tf.squeeze(raw_output)

        # print("Sampled actions: ", sampled_actions)
        
        # print("Amount of noise: ", self.noise)
        noise = np.random.normal(0, self.noise, self.num_actions)

        sampled_actions = sampled_actions.numpy() * (1 + noise)

        # print("Sampled actions noise: ", sampled_actions)

        if (len(known_keys) < self.num_actions):
            np.put(sampled_actions, np.arange(len(known_keys), self.num_actions), -np.inf*np.ones(int(self.num_actions-len(known_keys))))

        sampled_actions_probs = softmax(sampled_actions)

        # print("Sampled actions probs: ", sampled_actions_probs)


        return sampled_actions_probs, policy_type

    def memory(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    def remember(self, state, action, reward, next_state, done):
        # print("Action being stored: ", action)
        self.memory.store(state, action, reward, next_state, done)

    def has_data(self):
        """return true if agent has experiences from simulation
        """
        return (self.memory._mem_length > 0)

    # For distributed actors #
    def get_weights(self):
        return self.target_actor.get_weights()

    def set_weights(self, weights):
        self.target_actor.set_weights(weights)

    def set_priorities(self, indices, loss):
        # TODO implement this
        pass
