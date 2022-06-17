# Imports
import time
from agents.DeepQAgent import DeepQAgent
from agents.PPOAgent import BallerAgent


# Parameters
MAX_ITER = 100
DRAW = True
SAVE_IMAGES = True
LOAD_MODEL = True

agent = BallerAgent(DRAW, SAVE_IMAGES, LOAD_MODEL)
agent.train(1000, 100)

