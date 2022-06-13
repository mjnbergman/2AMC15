# Imports
import time
from agents.DeepQAgent import DeepQAgent


# Parameters
MAX_ITER = 100
DRAW = True
SAVE = True

agent = DeepQAgent()
agent.train()

