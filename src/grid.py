import matplotlib.pyplot as plt
from matplotlib import patches
import math

from src.square import Square

# def polar2cartesian(phi: float, r: float) -> tuple[float]:
#     """_summary_

#     Args:
#         phi (float): Rotation in radians
#         r (float): Distance form center

#     Returns:
#         tuple[float]: Direction vector corresponding
#     """
#     return (
#         r*math.cos(phi),
#         r*math.sin(phi)
#     )

class Grid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.obstacles = []
        self.goals = []
        self.robots = []

        self.fig = plt.figure()
        self.axes = self.fig.add_subplot(111)
        self.axes.set_aspect('equal', adjustable='box')
        self.axes.set_facecolor("black")
        self.axes.set_ylim([-1, height+1])
        self.axes.set_xlim([-1, width+1])
        patch = patches.Rectangle(
            xy=[0, 0],
            width=width, height=height,
            color="white"
        )
        self.axes.add_artist(patch)
        self.obstacle_patches = []
        self.goal_patches = []
        self.robot_patches = []
        self.robot_arrows = []

    def spawn_robots(self, robots, starting_positions, starting_rotations):
        self.robots = robots
        for i, robot in enumerate(robots):
            robot.spawn(self, *starting_positions[i], starting_rotations[i])
            robot_box = robot.history[-1]
            patch = patches.Circle(
                xy=[robot_box.x1 + 0.5*robot_box.x_size, robot_box.y1 + 0.5*robot_box.y_size],
                radius=0.5*robot.size,
                color="blue"
            )
            self.robot_patches.append(patch)
            self.axes.add_artist(patch)
            # arrow = patches.Arrow(
            #     robot_box.x1 + 0.5*robot_box.x_size,
            #     robot_box.y1 + 0.5*robot_box.y_size,
            #     *polar2cartesian(robot.rot, 0.5*robot_box.x_size),
            #     color="green"
            # )
            # self.robot_arrows.append(arrow)
            # self.axes.add_artist(arrow)

        for robot in robots:
            if self.is_blocked(robot):
                raise ValueError('Invalid starting pos, position is blocked!')

    def is_in_bounds(self, x, y, size_x, size_y):
        return x >= 0 and x + size_x <= self.width and y >= 0 and y + size_y <= self.height

    def put_obstacle(self, x, y, size_x, size_y):
        assert self.is_in_bounds(x, y, size_x, size_y)
        ob = Square(x, x + size_x, y, y + size_y)
        self.obstacles.append(ob)
        patch = patches.Rectangle(
            xy=[ob.x1, ob.y1],
            width=ob.x_size, height=ob.y_size,
            color="black"
        )
        self.obstacle_patches.append(patch)
        self.axes.add_artist(patch)

    def put_goal(self, x, y, size_x, size_y, resolution=0.5):
        assert self.is_in_bounds(x, y, size_x, size_y)
        num_tiles_x = int(size_x / resolution)
        size_x_tiles = size_x / num_tiles_x
        num_tiles_y = int(size_y / resolution)
        size_y_tiles = size_y / num_tiles_y

        for i in range(num_tiles_x):
            for j in range(num_tiles_y):
                goal = Square(
                    x + i*size_x_tiles, 
                    x + i*size_x_tiles + size_x_tiles, 
                    y + j*size_y_tiles, 
                    y + j*size_y_tiles + size_y_tiles
                )
                self.goals.append(goal)
                patch = patches.Rectangle(
                    xy=[goal.x1, goal.y1],
                    width=goal.x_size, height=goal.y_size,
                    color="orange"
                )
                self.goal_patches.append(patch)
                self.axes.add_artist(patch)

    def check_goals(self, robot):
        for i, goal in enumerate(self.goals):
            # if goal.distance(robot.bounding_box) <= 2**2:
            if goal.intersect(robot.bounding_box):
                self.goals.remove(goal)
                self.goal_patches[i].set_xy([-1000, -1000])
                self.goal_patches.remove(self.goal_patches[i])


    def is_blocked(self, robot):
        blocked_by_obstacle = any([ob.intersect(robot.bounding_box) for ob in self.obstacles])
        blocked_by_robot = any(
            [robot.bounding_box.intersect(other_robot.bounding_box) for other_robot in self.robots if
             other_robot.id != robot.id])
        return blocked_by_obstacle or blocked_by_robot

    def get_border_coords(self):
        return [0, self.width, self.width, 0, 0], [0, 0, self.height, self.height, 0]

    def plot_grid(self):
        for i, robot in enumerate(self.robots):
            robot_box = robot.history[-1]
            self.robot_patches[i].center = [robot_box.x1 + 0.5*robot_box.x_size, robot_box.y1 + 0.5*robot_box.y_size]
            # self.robot_arrows[i].x = robot_box.x1 + 0.5*robot_box.x_size
            # self.robot_arrows[i].y = robot_box.y1 + 0.5*robot_box.y_size
            # self.robot_arrows[i].dx, self.robot_arrows[i].dy = polar2cartesian(robot.rot, 0.5*robot_box.x_size)

            # self.robot_arrows[i].set_data(
            #     robot_box.x1 + 0.5*robot_box.x_size,
            #     robot_box.y1 + 0.5*robot_box.y_size,
            #     *polar2cartesian(robot.rot, 0.5*robot_box.x_size),
            #     color="green"
            # )
        plt.title('Battery levels: ' + '|'.join([str(round(robot.battery_lvl, 2)) for robot in self.robots]))
        plt.draw()
        plt.pause(0.0001)