import json
import cv2
from shapely.geometry import Polygon, MultiPolygon, LineString, GeometryCollection
import matplotlib.pyplot as plt
import numpy as np
import shutil
import os

from .plotting import plot_multipolygon, plot_polygon
from .utils import parse_roomsize, parse_polygons

class Grid:
    def __init__(self, configFile: str, robots: list, startingPos: list):
        """ Initialize environment

        Args:
            configFile (str): Filename for map json
            robots (list): List of robot objects to spawn
            startingPos (list): Corresponding starting positions (center)
        """        
        # Make folder for images
        if os.path.exists("images"):
            shutil.rmtree("images", ignore_errors=True)
        os.mkdir("images")

        # Load config JSON object
        with open(configFile) as file:
            config = json.load(file)
            
        # Keep track of all objects in environment
        self.roomsize = config["roomsize"]
        self.obstacles = GeometryCollection(
            parse_roomsize(config["roomsize"]) + \
            parse_polygons(config["obstacles"])
        )
        self.goals = MultiPolygon(parse_polygons(config["goals"]))
        self.totalGoalArea = self.goals.area
        self.death = MultiPolygon(parse_polygons(config["death"]))

        # Spawn robots
        self.moves = {}
        self.robots = robots
        for i, robot in enumerate(self.robots):
            robot.spawn(self, startingPos[i])
            self.moves[robot.id] = 0

        DPI = 10
        self.fig = plt.figure(figsize=(256/DPI, 256/DPI), dpi=DPI)
        self.axes = self.fig.add_axes([0., 0., 1., 1.])

        # Remove borders
        self.axes.set_xticklabels([])
        self.axes.set_xticks([])
        self.axes.set_yticklabels([])
        self.axes.set_yticks([])
        self.axes.margins(x=0, y=0)


    def plot_grid(self, resolution: int, draw: bool, save: bool) -> np.array:
        # Create figure 
        # Plot room background
        self.axes.set_xlim((-0.5, self.roomsize[0]+0.5))
        self.axes.set_ylim((-0.5, self.roomsize[1]+0.5))
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

        if draw:
            plt.draw()
            plt.pause(1)

        if save:
            cv2.imwrite(f"images/{max(self.moves.values())}.png", image[:, :, ::-1])

        return image

    
    

        

        