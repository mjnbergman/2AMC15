# Imports
import time
from src.robot import Robot
from src.utils import parse_config

# Load grid from file and spawn robot(s)
grid = parse_config('example.json')
grid.spawn_robots(
    [Robot(id=1, size=1, battery_drain_p=0.0, battery_drain_lam=10)], 
    [(0, 0)])

# Main loop
while True:
    # Show grid TODO: output to numpy image in real environment
    grid.plot_grid()

    # Stop simulation if all robots died or if everything is cleaned
    if all([not robot.alive for robot in grid.robots]) or len(grid.goals) == 0:
        break

    # Move robots one by one
    for robot in grid.robots:
        # To avoid deadlocks, only try to move alive robots
        if robot.alive:
            if not robot.move(p_random=0):
                robot.direction_vector = (1.1, 1.1)
grid.plot_grid()
time.sleep(3)