from copy import deepcopy
import numpy as np
import random
import enum

from src.square import Square
from src.grid import Grid


class Reward(enum.Enum):
    DEAD = -1000,
    GOAL = 1,
    TIME_PENALTY = -1


class Robot:
    def __init__(self, id: int, size: float = 1, battery_drain_p: float = 0,
                 battery_drain_lam: float = 0) -> None:
        """ Initializes robot

         Args:
            id (int): Unique identifier for robot.
            size (float): Physical size of robot, i.e. the area it cleans 
                (square bounding box is assumed).
            battery_drain_p (float): Probability that battery is drained.
            battery_drain_lambda (float): When battery is drained, it does so
                according to an exponential distribution with specified lambda
        """
        self.size = size
        self.id = id
        self.direction_vector = (0, 0)
        self.battery_drain_p = battery_drain_p
        self.battery_drain_lam = battery_drain_lam
        self.battery_lvl = 100
        self.alive = True
        self.time_penalty = -1

    def spawn(self, grid: Grid, start_x: float = 0, start_y: float = 0):
        self.pos = (start_x, start_y)
        self.bounding_box = Square(start_x, start_x + self.size, start_y, start_y + self.size)
        self.history = [self.bounding_box]
        self.grid = grid
        assert self.grid.is_in_bounds(start_x, start_y, self.size, self.size)

    def move(self, p_random: float = 0):

        reward = Reward.TIME_PENALTY

        # If we have a random move happen:
        if np.random.binomial(n=1, p=p_random) == 1:
            self.direction_vector = (random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1))
        # If there are no goals left, die:
        if len(self.grid.goals) == 0:
            self.alive = False
        # Cant move if we died:
        if self.alive:
            if self.direction_vector == (0, 0):  # Literally 0 speed so no movement.
                return False
            new_pos = tuple(np.array(self.pos) + self.direction_vector)
            # Temporarily set the new bounding box:
            new_box = deepcopy(self.bounding_box)
            new_box.update_pos(*new_pos)
            self.bounding_box = new_box
            if self.grid.is_blocked(self):
                return False
            elif not self.grid.is_in_bounds(new_pos[0], new_pos[1], self.size, self.size):
                return False
            else:
                do_battery_drain = np.random.binomial(1, self.battery_drain_p)
                if do_battery_drain == 1 and self.battery_lvl > 0:
                    self.battery_lvl -= (
                            np.random.exponential(self.battery_drain_lam) * abs(sum(self.direction_vector)))
                    if self.battery_lvl <= 0:
                        self.alive = False
                        self.battery_lvl = 0
                        return False
                del new_box
                self.pos = new_pos
                self.bounding_box.update_pos(*self.pos)
                self.history.append(self.bounding_box)
                # Check if in this position we have reached a goal:
                reward += Reward.GOAL * self.grid.check_goals(self)
                return True, reward
        else:
            return False, reward + Reward.DEAD
