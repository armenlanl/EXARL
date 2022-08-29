from deeprlalgo.utilities.data_structures.Config import Config
from deeprlalgo.agents.policy_gradient_agents.PPO import PPO

import exarl
from exarl.base.comm_base import ExaComm
from exarl.utils.globals import ExaGlobals
from exarl.utils.introspect import introspectTrace


class torch_ppo(exarl.ExaAgent):
    def __init__(self, env, is_learner):
        self.env = env
        self.is_learner = is_learner
        self.priority_scale = 0
        self.epsilon = 0
        self.step_number = 0

        batch_step_frequency = ExaGlobals.lookup_params('batch_step_frequency')
        assert batch_step_frequency <= 1, "Batch step frequency is not support by this agent" 

        self.train_count = 0
        self.train_frequency = ExaGlobals.lookup_params('train_frequency')
        if self.train_frequency == -1:
            self.train_frequency = ExaComm.agent_comm.size - ExaComm.num_learners
        if self.train_frequency < 1:
            self.train_frequency = 1
        

        self.nsteps = ExaGlobals.lookup_params('n_steps')
        self.episodes_per_round = ExaGlobals.lookup_params("batch_episode_frequency")
        self.config = Config()
        self.config.environment = self.env
        self.config.environment.spec.trials = 100
        self.config.seed = ExaGlobals.lookup_params('seed')
        self.config.use_GPU = ExaGlobals.lookup_params('use_GPU')
        self.config.randomise_random_seed = ExaGlobals.lookup_params('randomise_random_seed')
        self.config.hyperparameters = {
            "learning_rate": ExaGlobals.lookup_params("learning_rate"),
            "linear_hidden_units": ExaGlobals.lookup_params("linear_hidden_units"),
            "final_layer_activation": ExaGlobals.lookup_params("final_layer_activation"),
            "learning_iterations_per_round": self.train_frequency,
            "discount_rate": ExaGlobals.lookup_params("discount_rate"),
            "batch_norm": ExaGlobals.lookup_params("batch_norm"),
            "clip_epsilon": ExaGlobals.lookup_params("clip_epsilon"),
            "episodes_per_learning_round": ExaGlobals.lookup_params("batch_episode_frequency"),
            "normalise_rewards": ExaGlobals.lookup_params("normalise_rewards"),
            "gradient_clipping_norm": ExaGlobals.lookup_params("gradient_clipping_norm"),
            "mu": ExaGlobals.lookup_params("mu"),
            "theta": ExaGlobals.lookup_params("theta"),
            "sigma": ExaGlobals.lookup_params("sigma"),
            "epsilon_decay_rate_denominator": ExaGlobals.lookup_params("epsilon_decay_rate_denominator"),
            "clip_rewards": ExaGlobals.lookup_params("clip_rewards")
        }
        self.agent = PPO(self.config)  
        
        # For some envs this doesn't get set so we give an options via config
        average_score_required_to_win = ExaGlobals.lookup_params("average_score_required_to_win")
        if average_score_required_to_win != "None":
            self.agent.average_score_required_to_win = average_score_required_to_win

        self.many_episode_states = []
        self.many_episode_actions = []
        self.many_episode_rewards = []

        self.local_states = [[]]
        self.local_actions = [[]]
        self.local_rewards = [[]]

        self.exploration_epsilon = None
    
    def get_weights(self):
        return self.agent.policy_old.state_dict(), self.agent.policy_new.state_dict()
    
    def set_weights(self, weights):
        self.agent.policy_old.load_state_dict(weights[0])
        self.agent.policy_new.load_state_dict(weights[1])
        # Set this when we start a new batch of episodes
        self.exploration_epsilon = None
    
    def action(self, state):
        # Use our current episode to get epsilon for the batch of episodes
        # The workflow_episode isn't set in before set_weights
        # It only gets set once for the rollout
        if self.exploration_epsilon is None:
            self.exploration_epsilon =  self.agent.exploration_strategy.get_updated_epsilon_exploration({"episode_number": self.env.workflow_episode})
        return self.agent.experience_generator.pick_action(self.agent.policy_new, state, self.exploration_epsilon), 1

    def remember(self, state, action, reward, next_state, done):
        self.local_states[-1].append(state)
        self.local_actions[-1].append(action)
        self.local_rewards[-1].append(reward)
        self.step_number += 1

        if done or self.nsteps == self.step_number:
            self.local_states.append([])
            self.local_actions.append([])
            self.local_rewards.append([])
            self.step_number = 0

    def has_data(self):
        return len(self.local_rewards[-1]) > 0
    
    def generate_data(self):
        ret = (self.local_states, self.local_actions, self.local_rewards)
        self.local_states = [[]]
        self.local_actions = [[]]
        self.local_rewards = [[]]
        yield ret
   
    def train(self, batch):
        self.agent.many_episode_states.extend(batch[0])
        self.agent.many_episode_actions.extend(batch[1])
        self.agent.many_episode_rewards.extend(batch[2])
        self.train_count += 1

        if self.train_count % self.train_frequency == 0:
            self.agent.policy_learn()
            self.agent.update_learning_rate(self.agent.hyperparameters["learning_rate"], self.agent.policy_new_optimizer)
            self.agent.equalise_policies()
            
            # self.agent.save_and_print_result()
            self.agent.save_result()
            
            self.agent.many_episode_states = []
            self.agent.many_episode_actions = []
            self.agent.many_episode_rewards = []

    # Ignore
    def update_target(self):
        pass

    # This is for if you need to get some return from training back to an actor... Ignore for now
    def set_priorities(self, indices, loss):
        pass

