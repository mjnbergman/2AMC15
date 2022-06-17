from shapely.geometry import Point, LineString, GeometryCollection, MultiLineString, MultiPoint
from shapely.geometry.base import BaseMultipartGeometry
import numpy as np
import random

from .gym_env import GymEnv


class RobotAction:
    def __init__(self, direction_vector: float, p_random=0):
        self.direction_vector = Point(direction_vector)
        self.p_random = p_random


class Robot:
    def __init__(self, id: int, radius: float, color: str, batteryLevel: float):
        """Spawns robot instance on grid

        Args:
            id (int): Unique identifier for robot
            startPos (list): Starting position for robot in [x, y]
            radius (float): Radius for robot, i.e. total cleaning diameter is 
                2*`radius`
            grid (Grid): Grid on which to spawn robot
            batteryLevel (float): Starting amount of battery. i.e. distance
        """
        self.id = id
        self.radius = radius
        self.color = color
        self.batteryLevel = batteryLevel
        self.alive = True
        self.areaCleaned = 0

    def spawn(self, grid, startPos: list):
        """ Adds grid information to robot and sets startin location and boundary

        Args:
            grid (Grid): Grid to spawn robot onto
            startPos (list): [x, y] starting position of robot (centerpoint)
        """
        self.centerPoint = Point(startPos)
        self.boundaryPolygon = self.centerPoint.buffer(self.radius)
        self.grid = grid

    def move(self, action):

        directionPoint = Point(action.direction_vector) #.direction_vector

        # Determine if random move is taken
     #   if np.random.binomial(n=1, p=action.p_random) == 1:
     #       directionPoint = Point(
     #           random.uniform(-5, 5),
     #           random.uniform(-5, 5)
     #       )

        # Determine valid direction and new location
        directionPoint = self._valid_direction(directionPoint)
        newCenterPoint = Point(
            self.centerPoint.x + directionPoint.x,
            self.centerPoint.y + directionPoint.y
        )
        movementLine = LineString([self.centerPoint, newCenterPoint])
        movementPath = movementLine.buffer(self.radius)

        if self.alive:
            self.grid.moves[self.id] += 1

            # Determine death
            if movementPath.intersects(self.grid.death):
                self.alive = False

            # Update values
            self.centerPoint = newCenterPoint
            self.boundaryPolygon = self.centerPoint.buffer(self.radius)
            oldDirtyArea = self.grid.goals.area
            self.grid.goals -= movementPath
            newDirtyArea = self.grid.goals.area
            self.areaCleaned += (oldDirtyArea - newDirtyArea)

            # Update battery
            self.batteryLevel = max(0, self.batteryLevel - movementLine.length)
            if self.batteryLevel <= 0:
                self.alive = False

    def _valid_direction(self, directionPoint: Point, tol: float = 1e-2) -> Point:
        newCenterPoint = Point(
            self.centerPoint.x + directionPoint.x,
            self.centerPoint.y + directionPoint.y
        )

        movementPath = LineString([self.centerPoint, newCenterPoint])

        validMovements = movementPath - self.grid.obstacles.buffer(self.radius - tol)

        if validMovements.is_empty or self.centerPoint.distance(validMovements) > 1e-4:
            return Point(0, 0)
        if isinstance(validMovements, BaseMultipartGeometry):
            distanceToGeometries = np.array([
                self.centerPoint.distance(geometry)
                for geometry in validMovements.geoms
            ])
            closestGeometry = validMovements.geoms[np.argmin(distanceToGeometries)]
        else:
            closestGeometry = validMovements

        candidatePoints = [Point(coord) for coord in closestGeometry.coords]
        distanceToPoints = [
            self.centerPoint.distance(point)
            for point in candidatePoints
        ]
        newCenterPoint = candidatePoints[np.argmax(distanceToPoints)]

        return Point(
            newCenterPoint.x - self.centerPoint.x,
            newCenterPoint.y - self.centerPoint.y
        )
