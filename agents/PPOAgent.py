from continuous_environment.processor import RoombaProcessor, INPUT_SHAPE, WINDOW_LENGTH
from continuous_environment import Grid, Robot, RobotAction, GymEnv

import numpy as np
import gym
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras import layers

from collections import deque

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
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

        # Next, we build our self.policy_model. We use the same self.policy_model that was described by Mnih et al. (2015).
        self.input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE

        inputs = layers.Input(shape=self.input_shape)
        conv1 = layers.Conv2D(8, 3)(inputs)
        maxp1 = layers.MaxPooling2D(2)(conv1)
        conv2 = layers.Conv2D(16, 2)(maxp1)
        maxp2 = layers.MaxPooling2D(2)(conv2)
        flat = layers.Flatten(maxp2)

        actordense1 = layers.Dense(128, activation="tanh")(flat)
        actordense2 = layers.Dense(64, activation="softplus")(actordense1)

        criticdense1 = layers.Dense(128, activation="tanh")(flat)
        criticdense2 = layers.Dense(64, activation="tanh")(criticdense1)

        action = layers.Dense(2, activation="tanh")(1 + actordense2)
        critic = layers.Dense(1, activation="tanh")(criticdense2)

        self.policy_model = keras.Model(inputs=inputs, outputs=[action, critic])

        # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
        # even the metrics!
        self.memory = deque()  # SequentialMemory(limit=10000, window_length=WINDOW_LENGTH)
        self.processor = RoombaProcessor()

        self.policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                                           nb_steps=1000000)

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

    def train(self, nr_epochs, mini_batch_size, T, nr_actors):

        policy_optimizer = keras.optimizers.Adam(learning_rate=self.policy_learning_rate)
        value_optimizer = keras.optimizers.Adam(learning_rate=self.value_function_learning_rate)

        advantage_estimates = []
        value_diff_estimates = []

        epsilon = 0.05

        for epoch in range(nr_epochs):

            mini_batch_counter = 0
            mini_batch = []
            actor_episodes = []
            #  for actor in range(nr_actors):
            state = self.env.reset()
            advantage_estimate = 0
            value_diff_estimate = 0
            episode = []
            for time_step in range(T):
                action_probs, value_estimate = self.policy_model(state)

                chosen_action = np.random.beta(action_probs[0], action_probs[1])

                state, reward, done, info = self.env.step(chosen_action)

                action_distribution = tfp.distributions.Beta(action_probs[0], action_probs[1])

                self.memory.append(state, reward, done, value_estimate, chosen_action, action_distribution)

                if self.memory.size() > self.action_replay_length:
                    self.memory.popleft()

                if done:
                    break

            state_buffer, reward_buffer, done_buffer, \
            value_estimate_buffer, chosen_action, action_distribution = self._buffers_from_deque()

            for _ in range(self.model_train_iterations):
                kl = self._train_model(state_buffer, reward_buffer, done_buffer, value_estimate_buffer, chosen_action,
                                       action_distribution)

                if kl > 1.5 * target_kl:
                    # Early Stopping
                    break

    def _buffers_from_deque(self):
        tup_list = [list(x) for x in self.memory]
        return tup_list[:, 0], tup_list[:, 1], tup_list[:, 2], tup_list[:, 3], tup_list[:, 4], tup_list[:, 5]

    def _get_beta_log_probs(self, state, action):
        beta_parameters, _ = self.policy_model(state)
        beta_dist = tfp.distributions.Beta(beta_parameters[0], beta_parameters[1])
        return beta_dist.log_prob(action)


    def _train_model(self, state_buffer, reward_buffer, done_buffer, value_estimate_buffer, chosen_action,
                                       action_distribution):
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            ratio = tf.exp(
                self._get_beta_log_probs(state_buffer) - action_distribution.log_prob(chosen_action)
            )
            min_advantage = tf.where(
                advantage_buffer > 0,
                (1 + clip_ratio) * advantage_buffer,
                (1 - clip_ratio) * advantage_buffer,
            )

            policy_loss = -tf.reduce_mean(
                tf.minimum(ratio * advantage_buffer, min_advantage)
            )
            value_loss = tf.reduce_mean((return_buffer - critic(observation_buffer)) ** 2)

            total_loss = policy_loss + value_loss

        policy_grads = tape.gradient(total_loss, self.policy_model.trainable_variables)
        policy_optimizer.apply_gradients(zip(policy_grads, self.policy_model.trainable_variables))

        kl = tf.reduce_mean(
            logprobability_buffer
            - logprobabilities(actor(observation_buffer), action_buffer)
        )
        kl = tf.reduce_sum(kl)
        return kl
