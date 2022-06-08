# Imports
import time
from src.robot import Robot
from src.gym_environment import RoboEnvironment
from src.utils import parse_config

# Load grid from file and spawn robot(s)
env = RoboEnvironment("example.json")

while True:
    # Take a random action
    action = env.action_space[0]
    obs, reward, done, info = env.step(action)

    # Render the game
    env.render()

    if done:
        break

env.close()
#grid.plot_grid()
time.sleep(3)