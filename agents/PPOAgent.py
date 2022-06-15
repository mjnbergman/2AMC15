from continuous_environment.processor import RoombaProcessor, INPUT_SHAPE, WINDOW_LENGTH
from continuous_environment import Grid, Robot, RobotAction, GymEnv

import numpy as np
import gym
import tensorflow as tf

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
        self.policy_model = Sequential()
        #   if K.image_dim_ordering() == 'tf':
        # (width, height, channels)
        self.policy_model.add(Permute((2, 3, 1), input_shape=self.input_shape))
        #  elif K.image_dim_ordering() == 'th':
        #      # (channels, width, height)
        #      self.policy_model.add(Permute((1, 2, 3), input_shape=input_shape))
        #  else:
        #      raise RuntimeError('Unknown image_dim_ordering.')
        self.policy_model.add(Convolution2D(32, (8, 8), strides=(4, 4)))
        self.policy_model.add(Activation('relu'))
        self.policy_model.add(Convolution2D(64, (4, 4), strides=(2, 2)))
        self.policy_model.add(Activation('relu'))
        self.policy_model.add(Convolution2D(64, (3, 3), strides=(1, 1)))
        self.policy_model.add(Activation('relu'))
        self.policy_model.add(Flatten())
        self.policy_model.add(Dense(512))
        self.policy_model.add(Activation('relu'))
        self.policy_model.add(Dense(self.nb_actions))
        self.policy_model.add(Activation('linear'))
        self.policy_model.compile()

        # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
        # even the metrics!
        self.memory = SequentialMemory(limit=10000, window_length=WINDOW_LENGTH)
        self.processor = RoombaProcessor()

        self.policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                                           nb_steps=1000000)

        self.weights_filename = 'ppo_weights.h5f'
        self.checkpoint_weights_filename = 'ppo_weights.h5f_weights_{step}.h5f'
        self.log_filename = 'ppo_log.json'

        self.callbacks = [ModelIntervalCheckpoint(self.checkpoint_weights_filename, interval=250000)]
        self.callbacks += [FileLogger(self.log_filename, interval=100)]

        self.discount_factor = 0.9

    def train(self, nr_epochs, mini_batch_size, T, nr_actors):

        advantage_estimates = []
        value_diff_estimates = []

        epsilon = 0.05

        for epoch in range(nr_epochs):

            mini_batch_counter = 0
            mini_batch = []
            actor_episodes = []
            for actor in range(nr_actors):
                state = self.env.reset()
                advantage_estimate = 0
                value_diff_estimate = 0
                episode = []
                for time_step in range(T):
                    action_probs, value_estimate = self.policy_model(state)

                    state, reward, done, info = self.env.step(np.argmax(action_probs))

                    _, next_value_estimate = self.policy_model(state)

                    advantage_estimate += self.discount_factor ** time_step * \
                                          (reward - value_estimate + self.discount_factor * next_value_estimate)
                    value_diff_estimate += (advantage_estimate + value_estimate)

                    advantage_estimates.append(advantage_estimate)
                    value_diff_estimates.append(value_diff_estimate ** 2)

                    episode.append((state, np.argmax(action_probs), value_estimate))
                    mini_batch.append((advantage_estimate, value_diff_estimate, action_probs))
                    mini_batch_counter += 1

                    if mini_batch_counter > mini_batch_size:
                        mini_batch_counter = 0

                        with tf.gradient_tape() as tape:
                            ratio = self.policy_model(state) / action_probs
                            clip_loss = tf.reduce_mean(np.min(ratio * advantage_estimates,
                                                              tf.clip_by_value(ratio * advantage_estimates, 1 - epsilon,
                                                                               1 + epsilon)))
                            value_loss = tf.reduce_mean(value_diff_estimates)
                            total_loss = clip_loss - value_loss
                        tape.gradient(total_loss, self.policy_model.trainable_parameters)

                        advantage_estimates = []
                        value_diff_estimates = []

                    if done:
                        break
                actor_episodes.append(episode)
