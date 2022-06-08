from __future__ import annotations

class Square:
    def __init__(self, x1: float, x2: float, y1: float, y2: float) -> None:
        self.x1, self.x2, self.y1, self.y2 = x1, x2, y1, y2
        self.x_size = x2 - x1
        self.y_size = y2 - y1

    def intersect(self, other: Square) -> bool:
        """ Check if two squares intersect, i.e. contain eachother or overlap 
            edges

        Args:
            other (Square): Other square to check intersection with. Is usually 
                the robot bounding box.

        Returns:
            bool: intersection check    
        """
        intersecting = not (self.x2 <= other.x1 or self.x1 >= other.x2 or self.y2 <= other.y1 or self.y1 >= other.y2)
        inside = (other.x1 >= self.x1 and other.x2 <= self.x2 and other.y1 >= self.y1 and other.y2 <= self.y2)
        return intersecting or inside

    def update_pos(self, x: float, y: float) -> None:
        self.x1, self.x2, self.y1, self.y2 = x, x + self.x_size, y, y + self.y_size