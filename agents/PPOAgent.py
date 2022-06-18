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


class AdditionLayer(tf.keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(AdditionLayer, self).__init__()

    def call(self, inputs):
        return inputs + 1


class BallerAgent:
    def __init__(self, DRAW, SAVE, LOAD):

        # Get the environment and extract the number of actions.
        self.env = GymEnv(configFile="example-env.json",
                          robots=[
                              Robot(id=1, radius=0.1, color="blue", batteryLevel=100),
                              #     Robot(id=2, radius=1, color="green", batteryLevel=100)
                          ],
                          startingPos=[
                              [8, 5],
                              #        [2, 2]
                          ],
                          save=SAVE)
        self.SAVE = SAVE
        self.DRAW = DRAW
        self.LOAD = LOAD

        self.nb_actions = self.env.action_space.n

        self.input_shape = INPUT_SHAPE

        tf.compat.v1.enable_eager_execution()

        inputs = layers.Input(shape=self.input_shape)
        conv1 = layers.Conv2D(8, 3)(inputs)
        maxp1 = layers.MaxPooling2D(2)(conv1)
        conv2 = layers.Conv2D(16, 2)(maxp1)
        maxp2 = layers.MaxPooling2D(2)(conv2)
        flat = layers.Flatten()(maxp2)

        actordense1 = layers.Dense(128, activation="tanh")(flat)
        actordense2 = layers.Dense(64, activation="tanh")(actordense1)

        criticdense1 = layers.Dense(128, activation="tanh")(flat)
        criticdense2 = layers.Dense(64, activation="tanh")(criticdense1)

        action = layers.Dense(4, activation="softplus")(actordense2)

        # scaled_action = AdditionLayer()(action)

        # actor = layers.Dense(4, activation="relu")(scaled_action)
        critic = layers.Dense(1, activation="linear")(criticdense2)

        self.policy_model = tf.keras.Model(inputs=inputs, outputs=[action, critic])

        print(self.policy_model.summary())
        # plot_model(self.policy_model, "my_first_model_with_shape_info.png", show_shapes=True)

        # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
        # even the metrics!
        self.memory = deque()  # SequentialMemory(limit=10000, window_length=WINDOW_LENGTH)
        self.processor = RoombaProcessor()

        self.weights_filename = 'ppo_weights.h5f'
        self.checkpoint_weights_filename = 'ppo_weights.h5f_weights_{step}.h5f'
        self.log_filename = 'ppo_log.json'

        if self.LOAD:
            self.policy_model = tf.keras.models.load_model('ballerboi')

        self.discount_factor = 0.9

        self.policy_learning_rate = 3e-4
        self.value_function_learning_rate = 1e-3

        self.model_train_iterations = 50

        self.action_replay_length = 1000

        self.target_kl = 0.01

        self.clip_ratio = 0.2

        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=self.policy_learning_rate)

    def _scale_beta_to_direction_vector(self, direction_x, direction_y):
        return (direction_x - 1 / 2) * 8, (direction_y - 1 / 2) * 8

    def _scale_direction_vector_to_beta(self, direction_x, direction_y):
        return (direction_x / 8) + 1 / 2, (direction_y / 8) + 1 / 2

    def train(self, nr_epochs, T):

        advantage_estimates = []
        value_diff_estimates = []

        reward_tally = []

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
            episodic_reward = 0
            for time_step in range(T):
                # print(state.shape)

                if self.DRAW:
                    self.env.render()

                action_probs, value_estimate = self.policy_model(state)

                action_probs += 1

                action_distribution_1 = tfp.distributions.Beta(action_probs[0][0], action_probs[0][1])
                action_distribution_2 = tfp.distributions.Beta(action_probs[0][2], action_probs[0][3])

                chosen_action_x = action_distribution_1.sample()
                chosen_action_y = action_distribution_2.sample()

                # print("Baller moving to (", chosen_action_x, ", ", chosen_action_y, ")")

                chosen_action = RobotAction(
                    [self._scale_beta_to_direction_vector(chosen_action_x.numpy(), chosen_action_y.numpy())])

                state, reward, done, info = self.env.step(chosen_action)

                state = self.processor.process_observation(state)
                reward = self.processor.process_reward(reward)

                # print("Greyscale values: ", np.unique(state))

                episodic_reward += reward

                sampled_log_prob = action_distribution_1.log_prob(chosen_action_x) + action_distribution_2.log_prob(
                    chosen_action_y)

                #print("SAMPLED PROBABILITY: ", sampled_log_prob)

                self.memory.append((state, reward, done, value_estimate, chosen_action, sampled_log_prob))

                if len(self.memory) > self.action_replay_length:
                    self.memory.popleft()

                if done:
                    break
            reward_tally.append(episodic_reward)
            #print("EPISODIC REWARD: ", episodic_reward)
            state_buffer, reward_buffer, \
            value_estimate_buffer, chosen_action, action_distribution = self._buffers_from_deque()
            print(action_distribution)
            advantage_buffer = self._calculate_advantage_from_buffer(value_estimate_buffer, reward_buffer)
            return_buffer = self._calculate_return_from_buffer(reward_buffer)

            for _ in range(self.model_train_iterations):
                kl, p_loss, v_loss = self._train_model(state_buffer, advantage_buffer,
                                       chosen_action,
                                       action_distribution,
                                       return_buffer)
                print("Kullback-Leibler: ", kl)
                print("Policy loss: ", p_loss, ", Value Loss: ", v_loss)
                if kl > 1.5 * self.target_kl:
                # Early Stopping
                    print("EARLY STOPPING")
                    break
            self.policy_model.save("ballerboi")

    @tf.function
    def _calculate_advantage_from_buffer(self, value_estimate_buffer, reward_buffer):
        advantage_buffer = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)

        reward_tally = value_estimate_buffer[-1]
        for timestep in range(len(value_estimate_buffer) - 1, 0, -1):
            advantage_buffer = advantage_buffer.write(timestep, -value_estimate_buffer[timestep]+ reward_buffer[timestep] + reward_tally)
            reward_tally += tf.cast(reward_buffer[timestep], dtype=tf.dtypes.float32) * self.discount_factor
        stacked_buffer = advantage_buffer.stack()
        return (stacked_buffer - tf.reduce_mean(stacked_buffer)) / (tf.math.reduce_std(stacked_buffer) + 1e-10)

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
        return tf.stack(tup_list[:, 0]),\
        tf.cast(tf.stack(tup_list[:, 1]), dtype=tf.dtypes.float32),\
        tf.stack(tup_list[:, 3]), \
        tf.stack([tf.constant([x.direction_vector.x, x.direction_vector.y]) for x in tup_list[:, 4]]),\
        tf.stack((tup_list[:, 5]))

    @tf.function
    def _get_beta_log_probs(self, state, action):
       # print("In conversion")
        beta_parameters, _ = self.policy_model(state)

        beta_parameters += 1

      #  print("Parameters ", beta_parameters)

        beta_dist_1 = tfp.distributions.Beta(beta_parameters[0][0], beta_parameters[0][1])
        beta_dist_2 = tfp.distributions.Beta(beta_parameters[0][2], beta_parameters[0][3])

        dirx, diry = self._scale_direction_vector_to_beta(action[0], action[1])

     #   print("Scaled directions: ", dirx, diry)

        lp1 = beta_dist_1.log_prob(dirx)
        lp2 = beta_dist_2.log_prob(diry)

    #    print("LPs ", lp1, lp2)

        return lp1 + lp2

    @tf.function
    def _train_model(self, state_buffer, advantage_buffer, chosen_action,
                     action_distribution, return_buffer):

        kl_sum = 0.

        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            ratios = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
            print(ratios)
            for step in tf.range(len(state_buffer)):
        #        print("CONCATTING")
                # print("FIRST TERM")
                # print( self._get_beta_log_probs(state_buffer[step], chosen_action[step].direction_vector))
                # print("SECOND TERM________________________________________")
                # print(action_distribution[step])
                direction_tensor = chosen_action[step] #tf.constant([chosen_action[step][0],
                                                       #  chosen_action[step][1]],
                   # dtype=tf.dtypes.float32)
                kl = self._get_beta_log_probs(state_buffer[step], direction_tensor) - action_distribution[step]
                temporary_ratio = tf.exp(
                        kl
                    )
                kl_sum += -kl
        #        print("RATIO ", temporary_ratio, action_distribution[step], direction_tensor)
                ratios = ratios.write(step, temporary_ratio)

            ratios = ratios.stack()
            ratios = tf.clip_by_value(ratios, -1e2, 1e2)

            #ratios = np.array(ratios)
            #ratios = ratios[~np.isnan(ratios)]
            # advantage_buffer = advantage_buffer[~np.isnan(advantage_buffer)]
            # min_advantage = min_advantage[~np.isnan(min_advantage)]

         #   print(min_advantage)
            # print(advantage_buffer)
      #      print(ratios)
            # print(state_buffer.shape)
            #state_buffer = np.stack(state_buffer)
            state_buffer = tf.reshape(state_buffer, (state_buffer.shape[0], 84, 84, 1))

            #ratios, _ = self.policy_model(state_buffer)

            policy_loss = -tf.reduce_mean(
                tf.minimum(ratios * advantage_buffer,
                           tf.clip_by_value(ratios, self.clip_ratio, self.clip_ratio) * advantage_buffer)
            )

            #  print(state_buffer)
            _, value = self.policy_model(state_buffer)
            #  print("Value ", value)
            print(return_buffer)
            value_loss = tf.reduce_mean((return_buffer - value) ** 2)

            total_loss = policy_loss + value_loss
            print("Losses: ", total_loss, policy_loss, value_loss)
        policy_grads = tape.gradient(total_loss, self.policy_model.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(policy_grads, self.policy_model.trainable_variables))
        return (kl_sum/len(state_buffer)), policy_loss, value_loss
