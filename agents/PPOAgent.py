from continuous_environment.processor import RoombaProcessor, INPUT_SHAPE, WINDOW_LENGTH, INPUT_SHAPE_FIXED
from continuous_environment import Grid, Robot, RobotAction, GymEnv

import numpy as np
import gym
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras import layers

from tensorflow.keras.utils import plot_model

from collections import deque

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from tensorflow.keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


class BallerAgent:
    def __init__(self):

        # Get the environment and extract the number of actions.
        self.env = GymEnv(configFile="example-env.json",
                          robots=[
                              Robot(id=1, radius=0.5, color="blue", batteryLevel=100),
                              #     Robot(id=2, radius=1, color="green", batteryLevel=100)
                          ],
                          startingPos=[
                              [1, 1],
                              #        [2, 2]
                          ],
                          save=True)

        self.nb_actions = self.env.action_space.n

        self.input_shape = INPUT_SHAPE

        inputs = layers.Input(shape=self.input_shape)
        conv1 = layers.Conv2D(8, 3)(inputs)
        maxp1 = layers.MaxPooling2D(2)(conv1)
        conv2 = layers.Conv2D(16, 2)(maxp1)
        maxp2 = layers.MaxPooling2D(2)(conv2)
        flat = layers.Flatten()(maxp2)

        actordense1 = layers.Dense(128, activation="tanh")(flat)
        actordense2 = layers.Dense(64, activation="softplus")(actordense1)

        criticdense1 = layers.Dense(128, activation="tanh")(flat)
        criticdense2 = layers.Dense(64, activation="tanh")(criticdense1)

        action = layers.Dense(4, activation="tanh")(actordense2)

        scaled_action = layers.Lambda(lambda x: x + 1)(action)

        #actor = layers.Dense(4, activation="relu")(scaled_action)
        critic = layers.Dense(1, activation="tanh")(criticdense2)

        self.policy_model = tf.keras.Model(inputs=inputs, outputs=[scaled_action, critic])

        print(self.policy_model.summary())
        plot_model(self.policy_model, "my_first_model_with_shape_info.png", show_shapes=True)

        # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
        # even the metrics!
        self.memory = deque()  # SequentialMemory(limit=10000, window_length=WINDOW_LENGTH)
        self.processor = RoombaProcessor()

        self.weights_filename = 'ppo_weights.h5f'
        self.checkpoint_weights_filename = 'ppo_weights.h5f_weights_{step}.h5f'
        self.log_filename = 'ppo_log.json'

        self.callbacks = [ModelIntervalCheckpoint(self.checkpoint_weights_filename, interval=250000)]
        self.callbacks += [FileLogger(self.log_filename, interval=100)]

        self.discount_factor = 0.9

        self.policy_learning_rate = 3e-4
        self.value_function_learning_rate = 1e-3

        self.model_train_iterations = 50

        self.action_replay_length = 1000

        self.target_kl = 0.01

        self.clip_ratio = 0.05

        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=self.policy_learning_rate)

    def train(self, nr_epochs, T):

        advantage_estimates = []
        value_diff_estimates = []

        epsilon = 0.05

        for epoch in range(nr_epochs):

            mini_batch_counter = 0
            mini_batch = []
            actor_episodes = []
            #  for actor in range(nr_actors):
            state = self.env.reset()
            state = self.processor.process_observation(state)
            advantage_estimate = 0
            value_diff_estimate = 0
            episode = []
            for time_step in range(T):
                print(state.shape)
                action_probs, value_estimate = self.policy_model(state)
                print(action_probs)

                chosen_action_x = np.random.beta(action_probs[0][0], action_probs[0][1])
                chosen_action_y = np.random.beta(action_probs[0][2], action_probs[0][3])

                chosen_action = RobotAction([chosen_action_x, chosen_action_y])

                state, reward, done, info = self.env.step(chosen_action)

                state = self.processor.process_observation(state)
                reward = self.processor.process_reward(reward)

                action_distribution_1 = tfp.distributions.Beta(action_probs[0][0], action_probs[0][1])
                action_distribution_2 = tfp.distributions.Beta(action_probs[0][2], action_probs[0][3])

                sampled_log_prob = action_distribution_1.log_prob(chosen_action_x) + action_distribution_2.log_prob(
                    chosen_action_y)

                self.memory.append((state, reward, done, value_estimate, chosen_action, sampled_log_prob))

                if len(self.memory) > self.action_replay_length:
                    self.memory.popleft()

                if done:
                    break

            state_buffer, reward_buffer, done_buffer, \
            value_estimate_buffer, chosen_action, action_distribution = self._buffers_from_deque()
            advantage_buffer = self._calculate_advantage_from_buffer(value_estimate_buffer, reward_buffer)
            return_buffer = self._calculate_return_from_buffer(reward_buffer)

            for _ in range(self.model_train_iterations):
                kl = self._train_model(state_buffer, advantage_buffer, done_buffer, value_estimate_buffer,
                                       chosen_action,
                                       action_distribution,
                                       return_buffer)

                if kl > 1.5 * self.target_kl:
                    # Early Stopping
                    break

    def _calculate_advantage_from_buffer(self, value_estimate_buffer, reward_buffer):
        advantage_buffer = []
        reward_tally = value_estimate_buffer[-1]
        for timestep in range(len(value_estimate_buffer) - 1, 0, -1):
            advantage_buffer.append(-value_estimate_buffer[timestep] + reward_buffer[timestep] + reward_tally)
            reward_tally += reward_buffer[timestep] * self.discount_factor
        return (np.array(advantage_buffer) - np.mean(advantage_buffer)) / (np.std(advantage_buffer) + 1e-10)

    def _calculate_return_from_buffer(self, reward_buffer):

        discounted_reward = 0  # The discounted reward so far
        return_buffer = []
        # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
        # discounted return (think about why it would be harder starting from the beginning)
        for rew in reversed(reward_buffer):
            discounted_reward = rew + discounted_reward * self.discount_factor
            return_buffer.insert(0, discounted_reward)
        return return_buffer

    def _buffers_from_deque(self):
        tup_list = np.array([list(x) for x in self.memory])
        return tup_list[:, 0], tup_list[:, 1], tup_list[:, 2], tup_list[:, 3], tup_list[:, 4], tup_list[:, 5]

    def _get_beta_log_probs(self, state, action):
        beta_parameters, _ = self.policy_model(state)

        beta_dist_1 = tfp.distributions.Beta(beta_parameters[0][0], beta_parameters[0][1])
        beta_dist_2 = tfp.distributions.Beta(beta_parameters[0][2], beta_parameters[0][3])

        return beta_dist_1.log_prob(action.x) + beta_dist_2.log_prob(action.y)

    def _train_model(self, state_buffer, advantage_buffer, done_buffer, value_estimate_buffer, chosen_action,
                     action_distribution, return_buffer):
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            ratios = []
            for step in range(len(state_buffer)):
                ratios.append(tf.exp(
                    self._get_beta_log_probs(state_buffer[step], chosen_action[step].direction_vector) - action_distribution[step]
                ))

            min_advantage = tf.where(
                advantage_buffer > 0,
                (1 + self.clip_ratio) * advantage_buffer,
                (1 - self.clip_ratio) * advantage_buffer,
            )
            ratios = np.array(ratios)
            policy_loss = -tf.reduce_mean(
                tf.minimum(ratios * advantage_buffer, min_advantage)
            )
            #print(state_buffer.shape)
            state_buffer = np.stack(state_buffer)
            state_buffer = state_buffer.reshape(state_buffer.shape[0], 84, 84, 1)
            print(state_buffer)
            _, value = self.policy_model(state_buffer)
            print("Value ", value)
            print(return_buffer)
            value_loss = tf.reduce_mean((return_buffer - value) ** 2)

            total_loss = policy_loss + value_loss

        policy_grads = tape.gradient(total_loss, self.policy_model.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(policy_grads, self.policy_model.trainable_variables))

        # kl = tf.reduce_mean(
        #    logprobability_buffer
        #    - logprobabilities(actor(observation_buffer), action_buffer)
        # )
        # kl = tf.reduce_sum(kl)
        return 1
