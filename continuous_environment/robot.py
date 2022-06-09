from shapely.geometry import Point, LineString, GeometryCollection, MultiLineString, MultiPoint
from shapely.geometry.base import BaseMultipartGeometry
import numpy as np
import random

from .grid import Grid


class Robot:
    def __init__(self, id: int, radius: float, color: str):
        """Spawns robot instance on grid

        Args:
            id (int): Unique identifier for robot
            startPos (list): Starting position for robot in [x, y]
            radius (float): Radius for robot, i.e. total cleaning diameter is 
                2*`radius`
            grid (Grid): Grid on which to spawn robot
        """
        self.id = id
        self.radius = radius
        self.color = color
        self.batteryLevel = 100
        self.alive = True


    def spawn(self, grid: Grid, startPos: list):
        self.centerPoint = Point(startPos) 
        self.boundaryPolygon = self.centerPoint.buffer(self.radius)
        self.grid = grid


    def move(self, directionPoint: Point, pRandom: float):
        # Determine if random move is taken
        if np.random.binomial(n=1, p=pRandom) == 1:
            directionPoint = Point(
                random.uniform(-5, 5),
                random.uniform(-5, 5)
            )

        # Determine valid direction and new location
        directionPoint = self._valid_direction(directionPoint)
        newCenterPoint = Point(
            self.centerPoint.x + directionPoint.x,
            self.centerPoint.y + directionPoint.y
        )
        movementPath = LineString([self.centerPoint, newCenterPoint]).buffer(self.radius)


        if self.alive:
            self.grid.moves += 1

            # Determine death
            if movementPath.intersects(self.grid.death):
                self.alive = False
                return False
            
            self.centerPoint = newCenterPoint
            self.boundaryPolygon = self.centerPoint.buffer(self.radius)
            self.grid.goals -= movementPath



    
    def _valid_direction(self, directionPoint: Point, tol: float=1e-2) -> Point:
        newCenterPoint = Point(
            self.centerPoint.x + directionPoint.x,
            self.centerPoint.y + directionPoint.y
        )

        movementPath = LineString([self.centerPoint, newCenterPoint])

        validMovements = movementPath - self.grid.obstacles.buffer(self.radius-tol)

        if validMovements.is_empty or self.centerPoint.distance(validMovements) > 1e-4:
            print("No valid moves")
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

        

    # def _valid_direction(self, directionPoint: Point, tol: float=1e-3) -> Point:
    #     print(directionPoint)
    #     newCenterPoint = Point(
    #         self.centerPoint.x + directionPoint.x,
    #         self.centerPoint.y + directionPoint.y
    #     )

    #     movementPath = LineString([self.centerPoint, newCenterPoint])

    #     obstacleIntersection = movementPath.intersection(
    #         self.grid.obstacles.buffer(self.radius)
    #     )

    #     if obstacleIntersection.is_empty:
    #         return directionPoint
    #     if isinstance(obstacleIntersection, BaseMultipartGeometry):
    #         candidatePoints = []
    #         for geom in obstacleIntersection.geoms:
    #             candidatePoints += [Point(coord) for coord in geom.coords]
    #     else:
    #         candidatePoints = [Point(coord) for coord in obstacleIntersection.coords]


    #     validCenterPoints = self.grid.validArea \
    #         .buffer(-self.radius+tol, cap_style=2) \
    #         .buffer(self.radius-tol) \
    #         .buffer(-self.radius+tol) 

        
    #     candidatePoints = MultiPoint(candidatePoints) 
    #     candidatePoints = candidatePoints.intersection(validCenterPoints)
    #     if not isinstance(candidatePoints, MultiPoint):
    #         candidatePoints = MultiPoint([candidatePoints]) 

    #     if candidatePoints.is_empty:
    #         return Point(0, 0)
    #     else :
    #         distances = np.array([
    #             self.centerPoint.distance(candidatePoint)
    #             for candidatePoint in candidatePoints.geoms
    #         ])
    #         distances += (distances <= tol)*9999
    #         print(distances)
    #         validNewCenterPoint = candidatePoints.geoms[np.argmin(distances)]

    #     validMovementPath = LineString([self.centerPoint, validNewCenterPoint])
    #     if not validMovementPath.intersection(
    #         self.grid.obstacles.buffer(self.radius-tol)
    #     ).is_empty:
    #         return Point(0, 0)

    #     return Point(
    #         validNewCenterPoint.x - self.centerPoint.x,
    #         validNewCenterPoint.y - self.centerPoint.y
    #     )

