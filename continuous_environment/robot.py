from shapely.geometry import Point, LineString
from shapely.geometry.base import BaseMultipartGeometry
import numpy as np
from scipy.spatial import distance


class RobotAction:
    """ Object for tracking robot movements in gym environment
    """    
    def __init__(self, direction_vector: float, p_random: float=0):
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
        self.death_tile = False
        self.no_wall = True
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
        self.alive = True
        self.no_wall = True
        self.death_tile = False


    def move(self, action: RobotAction):
        """ Handle robot movement

        Args:
            action (RobotAction): Object with desired direction vector to move in
        """        
        directionPoint = Point(action.direction_vector)
        self.areaCleaned = 0

        # Determine valid direction and new location
        newDirectionPoint = self._valid_direction(directionPoint)

        # Check if obstacles are hit during movement
        self.no_wall = False
        self.death_tile = False

        # If valid movement is equal to desired, no wall must be hit
        if newDirectionPoint.x == directionPoint.x and newDirectionPoint.y == directionPoint.y:
            self.no_wall = True

        # Determine new center point after move and area passed over to clean
        newCenterPoint = Point(
            self.centerPoint.x + newDirectionPoint.x,
            self.centerPoint.y + newDirectionPoint.y
        )
        movementLine = LineString([self.centerPoint, newCenterPoint])
        movementPath = movementLine.buffer(self.radius)

        # Only handle if alive
        if self.alive:
            self.grid.moves[self.id] += 1

            # Determine death
            if movementPath.intersects(self.grid.death):
                self.alive = False
                self.death_tile = True

            # Update values
            self.centerPoint = newCenterPoint
            self.boundaryPolygon = self.centerPoint.buffer(self.radius)
            oldDirtyArea = self.grid.goals.area
            self.grid.goals -= movementPath
            newDirtyArea = self.grid.goals.area
            self.areaCleaned += (oldDirtyArea - newDirtyArea)

            # Update battery
            self.batteryLevel = max(0, self.batteryLevel - distance.euclidean(directionPoint.x, directionPoint.y))
            if self.batteryLevel <= 0:
                self.alive = False


    def _valid_direction(self, directionPoint: Point, tol: float = 1e-2) -> Point:
        """ Return valid direction vector in same direction, i.e. by handling 
            collisions. 

        Args:
            directionPoint (Point): Original desired direction
            tol (float, optional): Numerical tolerance. Defaults to 1e-2.

        Returns:
            Point: Valid direction vector in same direction as original
        """        
        # Determine new centerpoint if direction vector is added
        newCenterPoint = Point(
            self.centerPoint.x + directionPoint.x,
            self.centerPoint.y + directionPoint.y
        )

        # Get line between start and end point, i.e. the path the robot walks
        movementPath = LineString([self.centerPoint, newCenterPoint])

        # Remove obstacles along the movement path, leaving only valid positions
        # The obstacles are buffered to account for robot radius, as we consider
        # center point for movement path
        validMovements = movementPath - self.grid.obstacles.buffer(self.radius - tol)

        # If no moves arep ossible return empty
        if validMovements.is_empty or self.centerPoint.distance(validMovements) > 1e-4:
            return Point(0, 0)
        
        # Otherwise find part of valid movements that touches the robot, and 
        # then take the furthest point in that geometry as the new valid dir.
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
