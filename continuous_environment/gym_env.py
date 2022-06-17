import gym
from enum import IntEnum
from gym import Env, spaces
from gym.utils import seeding
import os
import shutil
import json
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, LineString, GeometryCollection
import matplotlib.pyplot as plt
from .utils import parse_roomsize, parse_polygons
from .plotting import plot_multipolygon, plot_polygon
import cv2
from copy import deepcopy

action_mapping = {0: [0, 1], 1: [1, 0], 2: [0, -1], 3: [-1, 0]}

class Reward(IntEnum):
    REWARD_PER_AREA = 1,
    TIME_PENALTY = -1,
    DEATH_PENALTY = -1


class GymEnv(Env):
    def __init__(self, configFile: str, robots: list, startingPos: list, save: bool):
        super(GymEnv, self).__init__()
        self._seed()

        # Make folder for images
        if os.path.exists("images"):
            shutil.rmtree("images", ignore_errors=True)
        os.mkdir("images")

        # Load config JSON object
        with open(configFile) as file:
            self.config = json.load(file)

        self.save = save

        # Keep track of all objects in environment
        self.roomsize = self.config["roomsize"]
        self.startingPos = startingPos
        self.initial_robots = robots
        self.robots = robots
        #self.action_space = spaces.Box(low=-5, high=5, shape=(len(self.robots),2),
        #                               dtype=np.uint8)
        self.action_space = spaces.Discrete(4)
        DPI = 10
        self.fig = plt.figure(figsize=(256 / DPI, 256 / DPI), dpi=DPI)
        self.axes = self.fig.add_axes([0., 0., 1., 1.])

        # Remove borders
        self.axes.set_xticklabels([])
        self.axes.set_xticks([])
        self.axes.set_yticklabels([])
        self.axes.set_yticks([])
        self.axes.margins(x=0, y=0)


        self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def render(self):

        plt.draw()
        plt.pause(0.1)

    def reset(self):
        print("RESET!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.obstacles = GeometryCollection(
            parse_roomsize(self.config["roomsize"]) + \
            parse_polygons(self.config["obstacles"])
        )
        self.goals = MultiPolygon(parse_polygons(self.config["goals"]))
        self.totalGoalArea = self.goals.area
        self.death = MultiPolygon(parse_polygons(self.config["death"]))

        # Spawn robots
        self.moves = {}
        self.robots = deepcopy(self.initial_robots)

        for i, robot in enumerate(self.robots):
            robot.spawn(self, self.startingPos[i])
            self.moves[robot.id] = 0

        # Create figure
        # Plot room background
        self.axes.set_xlim((-0.5, self.roomsize[0] + 0.5))
        self.axes.set_ylim((-0.5, self.roomsize[1] + 0.5))
        self.axes.set_facecolor("black")
        self.axes.fill(
            [0, 0, self.roomsize[0], self.roomsize[0], 0],
            [0, self.roomsize[1], self.roomsize[1], 0, 0],
            fc="white"
        )

        # Plot goal and death tiles
        plot_multipolygon(self.goals, "orange", self.axes)
        plot_multipolygon(self.death, "red", self.axes)

        # Plot obstacles
        for geometry in self.obstacles.geoms:
            if isinstance(geometry, MultiPolygon):
                plot_multipolygon(geometry, "black", self.axes)
            elif isinstance(geometry, Polygon):
                plot_polygon(geometry, "black", self.axes)
            elif isinstance(geometry, LineString):
                continue

        # Plot robots
        for robot in self.robots:
            if robot.alive:
                self.axes.fill(*robot.boundaryPolygon.exterior.xy, fc=robot.color)

        # Output to numpy array
        self.fig.canvas.draw()
        image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape((*self.fig.canvas.get_width_height(), 3))
        return image

    def step(self, actions):
        # Move robots
        for i, robot in enumerate(self.robots):
            print(f"Robot {robot.id} battery: {robot.batteryLevel}")
            print("Moving ", i, actions) #actions[i].direction_vector
            robot.move(actions)

        alive_vector = [robot.alive for robot in self.robots]
        reward_vector = [robot.areaCleaned * int(Reward.REWARD_PER_AREA) for robot in self.robots]

        # Create figure
        # Plot room background
        self.axes.set_xlim((-0.5, self.roomsize[0] + 0.5))
        self.axes.set_ylim((-0.5, self.roomsize[1] + 0.5))
        self.axes.set_facecolor("black")
        self.axes.fill(
            [0, 0, self.roomsize[0], self.roomsize[0], 0],
            [0, self.roomsize[1], self.roomsize[1], 0, 0],
            fc="white"
        )

        # Plot goal and death tiles
        plot_multipolygon(self.goals, "orange", self.axes)
        plot_multipolygon(self.death, "red", self.axes)

        # Plot obstacles
        for geometry in self.obstacles.geoms:
            if isinstance(geometry, MultiPolygon):
                plot_multipolygon(geometry, "black", self.axes)
            elif isinstance(geometry, Polygon):
                plot_polygon(geometry, "black", self.axes)
            elif isinstance(geometry, LineString):
                continue

        # Plot robots
        for robot in self.robots:
            if robot.alive:
                self.axes.fill(*robot.boundaryPolygon.exterior.xy, fc=robot.color)

        # Output to numpy array
        self.fig.canvas.draw()
        image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape((*self.fig.canvas.get_width_height(), 3))

        if self.save:
            cv2.imwrite(f"images/{max(self.moves.values())}.png", image[:, :, ::-1])

        print(np.all(~np.array(alive_vector)))
        print(alive_vector)

        return image, np.sum(reward_vector), np.all(~np.array(alive_vector)), {}
