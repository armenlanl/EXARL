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
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.optimizers import Adam

import exarl
from exarl.utils.globals import ExaGlobals
from exarl.agents.agent_vault._replay_buffer import ReplayBuffer
logger = ExaGlobals.setup_logger(__name__)

try:
    graph_size = ExaGlobals.lookup_params('graph_size')
except:
    graph_size = 20

def get_graph_adj(knownStates, state):
    adj_mat    = np.zeros([graph_size, graph_size])
    all_keys = sum(value != None for value in knownStates.values())
    # print("ALLKEYS: ", all_keys)

    # change to list knownStates.keys() where they are not equal to None
    graph_keys = [[state]]
    inc_keys   = [state]

    # Iterates 20 times
    for ii in range(graph_size):

        # Checks if # of included keys is equal to or greater than 20
        if len(inc_keys) >= graph_size:
            inc_keys = inc_keys[:graph_size]
            break

        if len(inc_keys) == all_keys:
            # Check to make sure tmp key is not equal to None
            tmp_keys = [x for x in knownStates.keys() if x not in inc_keys and knownStates[x] != None]
            if len(tmp_keys) > 0:
                if len(tmp_keys) + len(inc_keys) <= graph_size:
                    inc_keys += tmp_keys
                else:
                    inc_keys += tmp_keys[-(graph_size - len(inc_keys)):]
            break

        new_keys = []
        for key in graph_keys[ii]:
            tmp_keys = [x for x in knownStates[key].counts.keys() if (x not in inc_keys) and (x in knownStates.keys()) and (knownStates[x] != None)]
            
            inc_keys += tmp_keys
            new_keys += tmp_keys
            
        graph_keys.append(new_keys)
    
    # print("INC KEY: ", inc_keys)

    for ii, row_key in enumerate(inc_keys):
        for jj, col_key in enumerate(inc_keys):
            # print("ROW KEY: ", row_key)
            # print("COL KEY: ", col_key)
            if col_key in knownStates[row_key].counts.keys():
                adj_mat[ii,jj] = knownStates[row_key].counts[col_key]
    # print(adj_mat)
    return adj_mat

class KerasGraphTD3RL(exarl.ExaAgent):

    def __init__(self, env, is_learner, **kwargs):
        """ Define all key variables required for all agent """

        self.is_learner = is_learner
        # Get env info
        super().__init__(**kwargs)
        self.env = env
        # print("OBSERVATION SPACE SHAPE: ", env.observation_space.shape)
        self.num_states = env.observation_space.shape#[0]
        self.adj_shape = env.observation_space[0].shape
        # print("ADJ SPACE SHAPE: ", self.adj_shape)
        self.num_actions = env.action_space.shape[0]
        self.task_length = 5
        self.upper_bound = env.action_space.high
        self.lower_bound = env.action_space.low
        # print('upper_bound: ', self.upper_bound)
        # print('lower_bound: ', self.lower_bound)

        # Buffer
        self.buffer_counter = 0
        self.buffer_capacity = ExaGlobals.lookup_params('buffer_capacity')
        self.batch_size = ExaGlobals.lookup_params('batch_size')
        self.memory = ReplayBuffer(self.buffer_capacity, self.num_states, self.num_actions)
        # self.state_buffer = np.zeros((self.buffer_capacity, self.num_states))
        # self.action_buffer = np.zeros((self.buffer_capacity, self.num_actions))
        # self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        # self.next_state_buffer = np.zeros((self.buffer_capacity, self.num_states))
        # self.done_buffer = np.zeros((self.buffer_capacity, 1))
        self.per_buffer = np.ones((self.buffer_capacity, 1))

        # Used to update target networks
        self.tau = ExaGlobals.lookup_params('tau')
        self.gamma = ExaGlobals.lookup_params('gamma')

        # Setup Optimizers
        critic_lr = ExaGlobals.lookup_params('critic_lr')
        actor_lr = ExaGlobals.lookup_params('actor_lr')
        self.critic_optimizer1 = Adam(critic_lr, epsilon=1e-08)
        self.critic_optimizer2 = Adam(critic_lr, epsilon=1e-08)
        self.actor_optimizer = Adam(actor_lr, epsilon=1e-08)

        self.hidden_size = 56
        self.layer_std = 1.0 / np.sqrt(float(self.hidden_size))

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
        self.actor_update_freq = 2
        self.critic_update_freq = 2

        # Not used by agent but required by the learner class
        self.epsilon = ExaGlobals.lookup_params('epsilon')
        self.epsilon_min = ExaGlobals.lookup_params('epsilon_min')
        self.epsilon_decay = ExaGlobals.lookup_params('epsilon_decay')

        logger().info("TD3 buffer capacity {}".format(self.buffer_capacity))
        logger().info("TD3 batch size {}".format(self.batch_size))
        logger().info("TD3 tau {}".format(self.tau))
        logger().info("TD3 gamma {}".format(self.gamma))
        logger().info("TD3 critic_lr {}".format(critic_lr))
        logger().info("TD3 actor_lr {}".format(actor_lr))

    @tf.function
    def train_critic(self, states, actions, rewards, next_states, masks, masks_next):
        # states = states[0]
        # next_states = next_states[0]
        next_actions = self.target_actor(next_states, training=False)

        # print("next_actions SHAPE: ", next_actions.shape)
        # Add a little noise
        noise = np.random.normal(0, 0.2, self.num_actions)
        noise = np.clip(noise, -0.5, 0.5)
        next_actions = next_actions * (1 + noise)

        #Invalid action masking on next actions
        masks_next_complete = np.zeros(next_actions.shape, dtype=int)
        for i in range(len(masks_next)):
                mask_indices = np.array(masks_next[i])
                masks_next_complete[i][mask_indices] = 1
        
        next_actions = next_actions*masks_next_complete

        new_q1 = self.target_critic1([next_states, next_actions], training=False)
        new_q2 = self.target_critic2([next_states, next_actions], training=False)
        new_q = tf.math.minimum(new_q1, new_q2)
        # Bellman equation for the q value
        q_targets = rewards + self.gamma * new_q
        # Critic 1
        with tf.GradientTape() as tape:
            q_values1 = self.critic_model1([states, actions], training=True)
            td_errors1 = q_values1 - q_targets
            critic_loss1 = tf.reduce_mean(tf.math.square(td_errors1))
        gradient1 = tape.gradient(critic_loss1, self.critic_model1.trainable_variables)
        self.critic_optimizer1.apply_gradients(zip(gradient1, self.critic_model1.trainable_variables))

        # Critic 2
        with tf.GradientTape() as tape:
            q_values2 = self.critic_model2([states, actions], training=True)
            td_errors2 = q_values2 - q_targets
            critic_loss2 = tf.reduce_mean(tf.math.square(td_errors2))
        gradient2 = tape.gradient(critic_loss2, self.critic_model2.trainable_variables)
        self.critic_optimizer2.apply_gradients(zip(gradient2, self.critic_model2.trainable_variables))

    @tf.function
    def train_actor(self, states, masks):
        # Use Critic 1
        # states = states[0]
        with tf.GradientTape() as tape:
            actions = self.actor_model(states, training=True)
            # print("ACTIONS SHAPE: ", actions.shape)


            # Invalid action masking on next actions
            masks_next_complete = np.zeros(actions.shape, dtype=int)
            for i in range(len(masks)):
                mask_indices = np.array(masks[i])
                masks_next_complete[i][mask_indices] = 1

            actions = actions*masks_next_complete

            q_value = self.critic_model1([states, actions], training=True)
            loss = -tf.math.reduce_mean(q_value)
        gradient = tape.gradient(loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(gradient, self.actor_model.trainable_variables))

    def get_critic(self):
        # State as input
        state_input = tf.keras.layers.Input(shape=self.adj_shape + (1,))
        state_out = tf.keras.layers.Conv2D(32, kernel_size=5, activation="relu")(state_input)
        state_out = tf.keras.layers.Conv2D(16, kernel_size=5, activation="relu")(state_out)
        state_out = tf.keras.layers.Conv2D(4, kernel_size=5, activation="relu")(state_out)
        state_out = tf.keras.layers.Flatten()(state_out)
        state_out = tf.keras.layers.Dense(1028, activation="relu")(state_out)
        state_out = tf.keras.layers.Dense(256, activation="relu")(state_out)
        state_out = tf.keras.layers.Dense(64, activation="relu")(state_out)

        # Action as input
        action_input = tf.keras.layers.Input(shape=(self.num_actions))
        # action_out = tf.keras.layers.Dense(2 * self.num_actions, activation="relu")(action_input)

        # Both are passed through separate layer before concatenating
        concat = tf.keras.layers.Concatenate()([state_out, action_input])

        out = tf.keras.layers.Dense(8192, activation="relu")(concat)
        out = tf.keras.layers.Dense(4096, activation="relu")(out)
        out = tf.keras.layers.Dense(1024, activation="relu")(out)
        out = tf.keras.layers.Dense(256, activation="relu")(out)
        outputs = tf.keras.layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)
        model.summary()
        return model

    def get_actor(self):

        # MLP
        inputs = tf.keras.layers.Input(shape=self.adj_shape + (1,))
        #
        state_out = tf.keras.layers.Conv2D(32, kernel_size=5, activation="relu")(inputs)
        state_out = tf.keras.layers.Conv2D(16, kernel_size=5, activation="relu")(state_out)
        state_out = tf.keras.layers.Conv2D(4, kernel_size=5, activation="relu")(state_out)
        state_out = tf.keras.layers.Flatten()(state_out)
        state_out = tf.keras.layers.Dense(1028, activation="relu")(state_out)
        state_out = tf.keras.layers.Dense(256, activation="relu")(state_out)
        state_out = tf.keras.layers.Dense(64, activation="relu")(state_out)

        out = tf.keras.layers.Dense(self.hidden_size,
                                    kernel_initializer=RandomUniform(-self.layer_std, +self.layer_std),
                                    bias_initializer=RandomUniform(-self.layer_std, +self.layer_std))(state_out)
        out = tf.keras.layers.BatchNormalization()(out)
        out = tf.keras.layers.Activation(tf.nn.leaky_relu)(out)
        #
        outputs = tf.keras.layers.Dense(self.num_actions, activation="tanh",
                                        kernel_initializer=RandomUniform(-self.layer_std, +self.layer_std),
                                        bias_initializer=RandomUniform(-self.layer_std, +self.layer_std),
                                        use_bias=True)(out)

        # Rescale for tanh [-1,1]
        outputs = tf.keras.layers.Lambda(
            lambda x: ((x + 1.0) * (self.upper_bound - self.lower_bound)) / 2.0 + self.lower_bound)(outputs)

        model = tf.keras.Model(inputs, outputs)
        model.summary()
        return model

    @tf.function
    def soft_update(self, target_weights, weights):
        for (target_weight, weight) in zip(target_weights, weights):
            target_weight.assign(weight * self.tau + target_weight * (1.0 - self.tau))

    def update(self, state_batch, action_batch, reward_batch, next_state_batch):
        # print("SHAPE OF state_batch: ", state_batch.shape)
        # print("SHAPE OF state_batch[0]: ", state_batch[0,0].shape)
        # print("SHAPE OF state_batch[2]: ", type(state_batch[0,2]))
        # print("SHAPE OF next_state_batch: ", next_state_batch.shape)
        # print("Single batch: ", state_batch[0,1].shape)
        
        tf_state_batch = np.zeros(shape=(len(state_batch[:,0]),self.adj_shape[0],self.adj_shape[1]))
        tf_next_state_batch = np.zeros(shape=(len(next_state_batch[:,0]),self.adj_shape[0],self.adj_shape[1]))
        masks = []
        masks_next = []

        for i in range(len(state_batch[:,0])):
            tf_state_batch[i] = state_batch[i,0]
            tf_next_state_batch[i] = tf_next_state_batch[i,0]
            known_keys = [x for x in state_batch[i,2].keys() if state_batch[i,2][x] != None]
            known_keys_next = [x for x in next_state_batch[i,2].keys() if next_state_batch[i,2][x] != None]
            masks.append(known_keys)
            masks_next.append(known_keys_next)
            print(action_batch[0])
            print("Batch idx: ", i, " Values that don't equal zero:", sum(value != 0 for value in action_batch[i]))

        tf_state_batch = tf.convert_to_tensor(tf_state_batch, dtype=tf.float32)
        tf_next_state_batch = tf.convert_to_tensor(tf_next_state_batch, dtype=tf.float32)

        

        self.train_critic(tf_state_batch, action_batch, reward_batch, tf_next_state_batch, masks, masks_next)
        self.train_actor(tf_state_batch, masks)

    def _convert_to_tensor(self, state_batch, action_batch, reward_batch, next_state_batch, terminal_batch):
        for i in range(len(state_batch[:,0])):
            state_batch[i,0] = tf.convert_to_tensor(state_batch[i,0], dtype=tf.float32)
        # state_batch = tf.convert_to_tensor(state_batch[:,0], dtype=tf.float32)
        action_batch = tf.convert_to_tensor(action_batch, dtype=tf.float32)
        reward_batch = tf.convert_to_tensor(reward_batch, dtype=tf.float32)
        for i in range(len(next_state_batch[:,0])):
            next_state_batch[i,0] = tf.convert_to_tensor(next_state_batch[i,0], dtype=tf.float32)
        # next_state_batch[:,0] = tf.convert_to_tensor(next_state_batch[:,0], dtype=tf.float32)
        terminal_batch = tf.convert_to_tensor(terminal_batch, dtype=tf.float32)
        return state_batch, action_batch, reward_batch, next_state_batch, terminal_batch

    def generate_data(self):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = \
            self._convert_to_tensor(*self.memory.sample_buffer(self.batch_size))
        yield state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def train(self, batch):
        """ Method used to train """
        self.ntrain_calls += 1
        self.update(batch[0], batch[1], batch[2], batch[3])

    def update_target(self):
        if self.ntrain_calls % self.actor_update_freq == 0:
            self.soft_update(self.target_actor.variables, self.actor_model.variables)
        if self.ntrain_calls % self.critic_update_freq == 0:
            self.soft_update(self.target_critic1.variables, self.critic_model1.variables)
            self.soft_update(self.target_critic2.variables, self.critic_model2.variables)


    def action(self, state):
        taskList = []
        taskList.append(state[1])
        known_keys = [x for x in state[2].keys() if state[2][x] != None]
        
        next_adj = state[0]
        for _ in range(self.task_length-1):
            returned_action, _ = self.mini_action(next_adj, known_keys)
            next_adj = get_graph_adj(state[2], returned_action)
            taskList.append(returned_action)

        policy_type = 1

        print("Current State: ", state[1])
        print("Known Keys: ", known_keys)
        print("Task List: ", taskList)

        return taskList, policy_type
    
    def mini_action(self, state, known_keys):
        """ Method used to provide the next action using the target model """
        # print("State: ", state)
        # dictator = state[2]
        # state = state[0]
        
        # print("ACTOR CHECK: ", dictator[5050])

        tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        sampled_actions = tf.squeeze(self.actor_model(tf_state))
        # print("Sampled action output from the actor: ", sampled_actions)

        noise = np.random.normal(0, 0.1, self.num_actions)
        sampled_actions = sampled_actions.numpy() * (1 + noise)
        policy_type = 1

        # print("Sampled noisey action output from the actor: ", sampled_actions)

        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, self.lower_bound, self.upper_bound)

        mask_indices = np.array(known_keys)
        mask = np.zeros(100000, dtype=int)
        mask[mask_indices] = 1
        legal_action = legal_action*mask

        final_action = np.array(legal_action).argmax()
        # print("Final legal action: ", final_action)

        # return [np.squeeze(legal_action)], [np.squeeze(noise)]
        # return [np.squeeze(legal_action)], policy_type
        return final_action, policy_type

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
        print("Action being stored: ", action)
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
