from continuous_environment.processor import RoombaProcessor, INPUT_SHAPE, WINDOW_LENGTH
from continuous_environment import Grid, Robot, RobotAction, GymEnv

import numpy as np
import gym

import tensorflow

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from tensorflow.keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


class DeepQAgent:
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

        # Next, we build our self.model. We use the same self.model that was described by Mnih et al. (2015).
        self.input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
        self.model = Sequential()
        #   if K.image_dim_ordering() == 'tf':
        # (width, height, channels)
        self.model.add(Permute((2, 3, 1), input_shape=self.input_shape))
        #  elif K.image_dim_ordering() == 'th':
        #      # (channels, width, height)
        #      self.model.add(Permute((1, 2, 3), input_shape=input_shape))
        #  else:
        #      raise RuntimeError('Unknown image_dim_ordering.')
        self.model.add(Convolution2D(32, (8, 8), strides=(4, 4)))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(64, (4, 4), strides=(2, 2)))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(64, (3, 3), strides=(1, 1)))
        self.model.add(Activation('relu'))
        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dense(self.nb_actions))
        self.model.add(Activation('linear'))
        print(self.model.summary())

        # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
        # even the metrics!
        self.memory = SequentialMemory(limit=10000, window_length=WINDOW_LENGTH)
        self.processor = RoombaProcessor()

        self.policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                                           nb_steps=1000000)

        self.dqn = DQNAgent(model=self.model, nb_actions=self.nb_actions, policy=self.policy, memory=self.memory,
                            processor=self.processor, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
                            train_interval=4, delta_clip=1.)
        self.dqn.compile(Adam(lr=.00025), metrics=['mae'])

        self.weights_filename = 'dqn_weights.h5f'
        self.checkpoint_weights_filename = 'dqn_weights.h5f_weights_{step}.h5f'
        self.log_filename = 'dqn_log.json'

        self.callbacks = [ModelIntervalCheckpoint(self.checkpoint_weights_filename, interval=250000)]
        self.callbacks += [FileLogger(self.log_filename, interval=100)]

    def train(self):
        # Okay, now it's time to learn something! We capture the interrupt exception so that training
        # can be prematurely aborted. Notice that now you can use the built-in Keras callbacks!

        self.dqn.fit(self.env, callbacks=self.callbacks, nb_steps=17500, log_interval=1000)

        # After training is done, we save the final weights one more time.
        self.dqn.save_weights(self.weights_filename, overwrite=True)
