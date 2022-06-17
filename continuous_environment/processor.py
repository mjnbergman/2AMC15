from PIL import Image
import numpy as np
from rl.core import Processor

INPUT_SHAPE = (84, 84, 1)
INPUT_SHAPE_FIXED = (84, 84)
INPUT_SHAPE_BATCH = (1, 84, 84, 1)
WINDOW_LENGTH = 4


class RoombaProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 3  # (height, width, channel)
       # print(observation)
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE_FIXED).convert('L')  # resize and convert to grayscale
        processed_observation = np.array(img).reshape(INPUT_SHAPE_BATCH)
       # print(processed_observation.reshape((INPUT_SHAPE_BATCH)))
        assert processed_observation.shape == INPUT_SHAPE_BATCH
        return processed_observation.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        #return np.clip(reward, -1., 1.)
        return reward
