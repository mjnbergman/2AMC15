# Imports
import time
#from agents.DeepQAgent import DeepQAgent
from agents.PPOAgent import BallerAgent
from continuous_environment.gym_env import GymEnv
from continuous_environment.robot import Robot

# Parameters
MAX_ITER = 100
DRAW = True
SAVE_IMAGES = True
LOAD_MODEL = False

#agent = BallerAgent(DRAW, SAVE_IMAGES, LOAD_MODEL)
#agent.train(1000, 100)

env = GymEnv(configFile="example-env.json",
                  robots=[
                      Robot(id=1, radius=0.1, color="blue", batteryLevel=100),
                      #     Robot(id=2, radius=1, color="green", batteryLevel=100)
                  ],
                  startingPos=[
                      [8, 5],
                      #        [2, 2]
                  ],
                  save=SAVE_IMAGES)

env.reset()
action = env.action_space.sample()
print(action, env.step(action))

