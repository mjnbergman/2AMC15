###
#  gym_env: 
#  
#  GymEnv: custom Gym compliant environment
#   - render function that renders the environment
#   - reset function that resets it
#   - step function that takes one step updates environment and returns reward
#  
#  Reward: Reward map for environment


from enum import IntEnum
from gym import Env, spaces,GoalEnv
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
import time

RESOLUTION = 128


class Reward(IntEnum):
    """ Reward map for gym environment
    """    
    REWARD_PER_AREA = 100,
    TIME_PENALTY = -1,
    DEATH_PENALTY = -100,
    WALL_PENALTY = -1


class GymEnv(Env):
    def __init__(self, configFile: str, robots: list, startingPos: list, save: bool):
        """ Created gym environment

        Args:
            configFile (str): Filename of map json, e.g. `example-env.json`
            robots (list): List of Robot objects to spawn on grid
            startingPos (list): Corresponding starting positions of robots.
                Corresponds to center of robot in [x, y] format
            save (bool): Whether images of the environment need to be saved
        """        
        super(GymEnv, self).__init__()
        self._seed()

        # # Make folder for images
        # Make folder for images
        if os.path.exists("images"):
            shutil.rmtree("images", ignore_errors=True)
        os.mkdir("images")

        # Load config JSON object
        with open(configFile) as file:
            self.config = json.load(file)

        self.save = save
        
        self.percentage_cleaned = []
        
        # Keep track of all objects in environment
        self.roomsize = self.config["roomsize"]
        self.startingPos = startingPos
        self.initial_robots = robots
        self.robots = robots
        self.observation_space = spaces.Box(0, 255, [RESOLUTION, RESOLUTION, 3], dtype=np.uint8)
        self.action_space = spaces.Box(low=-5, high=5, shape=(2,),
                                       dtype=np.float32)

        DPI = 10
        self.fig = plt.figure(figsize=(RESOLUTION / DPI, RESOLUTION / DPI), dpi=DPI)
        self.axes = self.fig.add_subplot(111)

        # Remove borders
        self.axes.set_xticklabels([])
        self.axes.set_xticks([])
        self.axes.set_yticklabels([])
        self.axes.set_yticks([])
        self.axes.margins(x=0, y=0)

        #self.reward_fig = plt.figure(figsize=(1024 / DPI, RESOLUTION / DPI), dpi=DPI)
        # self.reward_axes = self.fig.add_subplot(122)
        self.reward_tally = []
        self.reward_window_size = 10

        self.reset()


    def _seed(self, seed=None):
        """ Seed gym environment 
        """        
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def render(self):
        """ Renders map in current state and running average of 
            rewards over time

        """     
        plt.figure(1)
        plt.draw()
        plt.pause(0.1)

        if len(self.reward_tally) > 0:
            running_average = np.convolve(
                self.reward_tally,
                np.ones(self.reward_window_size) / self.reward_window_size, 
                mode='valid'
            )
            plt.plot(range(len(running_average)), running_average)
            plt.title(time.ctime())
            plt.savefig("loss.jpg")
            plt.pause(0.001)


    def reset(self):
        """ Reset the gym environment for the next iteration 

        Returns: 
            environment image
        """     
        plt.figure(1)
        
        print("RESET!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        
        # Make folder for images
        if os.path.exists("images"):
            shutil.rmtree("images", ignore_errors=True)
        os.mkdir("images")

        self.obstacles = GeometryCollection(
            parse_roomsize(self.config["roomsize"]) + \
            parse_polygons(self.config["obstacles"])
        )

        # Read out goals, calculate goal area and read out death areas
        self.goals = MultiPolygon(parse_polygons(self.config["goals"]))
        self.totalGoalArea = self.goals.area
        self.death = MultiPolygon(parse_polygons(self.config["death"]))
       
        self.moves = {}
        
        # Repawn robots
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
            if isinstance(geometry, Polygon):
                plot_polygon(geometry, "black", self.axes)
            elif isinstance(geometry, MultiPolygon):
                plot_multipolygon(geometry, "black", self.axes)
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
        """ Move the robot in the direction of the action. Checks
        if move is valid and if not corrects it. Returns new image, reward and if game ended
        
        Args:
            actions (tuple): the movement of the robot

        Returns: 
            environment image, reward, 
        """
        for i, robot in enumerate(self.robots):
            robot.move(actions)

        
        # Update robot status and emit rewards
        alive_vector = [robot.alive for robot in self.robots]
        reward_vector = [robot.areaCleaned * int(Reward.REWARD_PER_AREA)
                         + int(not robot.no_wall) * int(Reward.WALL_PENALTY)
                         + int(robot.death_tile) * int(Reward.DEATH_PENALTY) for robot in self.robots]

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
            if isinstance(geometry, Polygon):
                plot_polygon(geometry, "black", self.axes)
            elif isinstance(geometry, MultiPolygon):
                plot_multipolygon(geometry, "black", self.axes)
            elif isinstance(geometry, LineString):
                continue

        # Plot robots
        for robot in self.robots:
            if robot.alive:
                self.axes.fill(*robot.boundaryPolygon.exterior.xy, fc=robot.color)

        # Output to numpy array
        self.fig.canvas.draw()
        plt.plot()
        image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape((*self.fig.canvas.get_width_height(), 3))

        if self.save:
            cv2.imwrite(f"images/{max(self.moves.values())}.png", image[:, :, ::-1])

        self.reward_tally.append(np.sum(reward_vector))

        # If robots are not alive or floor cleaned 95%, print percentage cleaned this run and store it
        if self.robots[0].grid.goals.area/self.totalGoalArea < .05 or bool(np.all(~np.array(alive_vector))) == True:
            print(max(self.moves.values()), self.robots[0].grid.goals.area/self.totalGoalArea)
            self.percentage_cleaned.append(self.robots[0].grid.goals.area/self.totalGoalArea)
        
        # If floor cleaned 95% quit
        if self.robots[0].grid.goals.area/self.totalGoalArea < .05:
            return image, np.sum(reward_vector), True, {}

        return image, np.sum(reward_vector), bool(np.all(~np.array(alive_vector))), {}
