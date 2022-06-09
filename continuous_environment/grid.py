import json
import cv2
from shapely.geometry import Polygon, MultiPolygon, LineString, GeometryCollection
import matplotlib.pyplot as plt
import numpy as np
import shutil
import os

class Grid:
    def __init__(self, configFile: str, robots: list, startingPos: list):
        # Make folder for images
        if os.path.exists("images"):
            shutil.rmtree("images", ignore_errors=True)
        os.mkdir("images")

        # Load config JSON object
        with open(configFile) as file:
            config = json.load(file)
            
        # Keep track of all objects in environment
        self.moves = 0
        self.roomsize = config["roomsize"]
        self.roomPolygon = Polygon([
            [0, 0],
            [0, self.roomsize[1]],
            [self.roomsize[0], self.roomsize[1]],
            [self.roomsize[0], 0]
        ])
        self.obstacles = GeometryCollection(
            self._parse_roomsize(config["roomsize"]) + \
            self._parse_polygons(config["obstacles"])
        )
        self.validArea = self.roomPolygon - \
            MultiPolygon(self._parse_polygons(config["obstacles"]))
        self.goals = MultiPolygon(self._parse_polygons(config["goals"]))
        self.death = MultiPolygon(self._parse_polygons(config["death"]))

        # Spawn robots
        self.robots = robots
        for i, robot in enumerate(self.robots):
            robot.spawn(self, startingPos[i])

        


    


    def plot_grid(self, resolution: int, draw: bool, save: bool) -> np.array:
        # Create figure 
        DPI = 10
        self.fig = plt.figure(figsize=(resolution/DPI, resolution/DPI), dpi=DPI)
        self.axes = self.fig.add_axes([0.,0.,1.,1.])

        # Remove borders
        self.axes.set_xticklabels([])
        self.axes.set_xticks([])
        self.axes.set_yticklabels([])
        self.axes.set_yticks([])
        self.axes.margins(x=0, y=0)

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
        self._plot_multipolygon(self.goals, "orange", self.axes)
        self._plot_multipolygon(self.death, "red", self.axes)

        # Plot obstacles
        for geometry in self.obstacles.geoms:
            if isinstance(geometry, MultiPolygon):
                self._plot_multipolygon(geometry, "black", self.axes)
            elif isinstance(geometry, Polygon):
                self._plot_polygon(geometry, "black", self.axes)
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
            plt.pause(0.0001)

        if save:
            cv2.imwrite(f"images/{self.moves}.png", image[:, :, ::-1])

        return image


    
    def _plot_multipolygon(self, multiPolygon: MultiPolygon, color: str, axes):
        """Plots MultiPolygon in `color` to `axes`

        Args:
            multiPolygon (MultiPolygon): Shapes to plot
            color (str): Color to plot in, e.g. "blue"
            axes (_type_): Matplotlib axes to plot to
        """
        for polygon in multiPolygon.geoms:
            axes.fill(*polygon.exterior.xy, fc=color)

    def _plot_polygon(self, polygon: Polygon, color: str, axes):
        """Plots Polygon in `color` to `axes`

        Args:
            polygon (MultiPolygon): Shape to plot
            color (str): Color to plot in, e.g. "blue"
            axes (_type_): Matplotlib axes to plot to
        """
        axes.fill(*polygon.exterior.xy, fc=color)


    def _parse_roomsize(self, roomsize: list) -> list:
        """ Parses room size to LineString representing outer bounding box. Used
            to parse room size.

        Args:
            roomsize (list[width, height]): Room size laoded form config file

        Returns:
            list[LineString]: List of length one containing LineString for outer
                boundary of room.
        """  
        return [LineString([
            (0, 0),                      # Bottom left
            (0, roomsize[1]),            # Top left
            (roomsize[0], roomsize[1]),  # Top right
            (roomsize[0], 0),            # Bottom right
            (0, 0)
        ])]      


    def _parse_polygons(self, multiPolygonCoords: list) -> list:
        """ Parses nested structure in config file to list of Polygons. Used to 
            parse goals, deaths, and walls.

        Args:
            multiPolygonCoords (list[list[list]]): 
                Structure: [ [[0, 0], [1, 0], ...] , ...]

        Returns:
            list[Polygon]: List of polygons
        """
        return [Polygon(polygonCoords) for polygonCoords in multiPolygonCoords]

        

        