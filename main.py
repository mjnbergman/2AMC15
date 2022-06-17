# Imports
import time
from agents.DeepQAgent import DeepQAgent
from agents.PPOAgent import BallerAgent


# Parameters
MAX_ITER = 100
DRAW = True
SAVE = True

agent = BallerAgent()
agent.train(1000, 100)

