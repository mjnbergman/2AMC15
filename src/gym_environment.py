import gym
from gym import Env
from utils import parse_config
from robot import Robot
import numpy as np


class RoboEnvironment(Env):

    def __init__(self, grid_location):
        super(RoboEnvironment, self).__init__()
        self.previous_location = grid_location
        self.grid = None
        self.dead = False
        self.initialize_environment(grid_location)
        self.action_space = [(1.1, 1.1)]

    def initialize_environment(self, location):
        self.grid = parse_config(location)
        self.initialize_robots()

    def initialize_robots(self, nr_robot=1):
        self.grid.spawn_robots(
            [Robot(id=i, size=1, battery_drain_p=0.0, battery_drain_lam=10) for i in range(nr_robot)],
            [(np.random.uniform(0, self.grid.width), np.random.uniform(0, self.grid.height)) for i in range(nr_robot)])

    def reset(self):
        self.initialize_environment(self.previous_location)
        return self.grid.fig2rgb_array()

    def step(self, action):
        # Stop simulation if all robots died or if everything is cleaned
        if all([not robot.alive for robot in self.grid.robots]) or len(self.grid.goals) == 0:
            self.dead = True

        total_reward = 0

        # Move robots one by one
        for robot in self.grid.robots:
            # To avoid deadlocks, only try to move alive robots
            if robot.alive:
                robot.direction_vector = action
                alive, reward = robot.move(p_random=0)
                total_reward += reward

        return self.grid.fig2rgb_array(), total_reward, self.dead, []

    def render(self):
        self.grid.plot_grid()
