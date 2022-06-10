# Imports
import time
from continuous_environment import Grid, Robot

# Define environment and robots
grid = Grid(
    configFile="example-env.json", 
    robots=[
        Robot(id=1, radius=0.5, color="blue")
    ],
    startingPos=[
        [1, 1]
    ]
)

# Parameters
MAX_ITER = 100
DRAW = False
SAVE = True

# Main loop
start_time = time.time() # Measure time
while grid.moves < MAX_ITER:
    # Check if environment is still active
    if all([not robot.alive for robot in grid.robots]):
        print(f"{grid.moves} moves completed before death")
        break

    # Get environment image for robot
    image = grid.plot_grid(resolution=256, draw=DRAW, save=SAVE)

    # Move robots 
    for robot in grid.robots:
        if robot.alive:
            robot.move(None, 1)

# Final state
image = grid.plot_grid(resolution=256, draw=DRAW, save=SAVE)
end_time = time.time() # Measure time

print(f"Time taken: {end_time - start_time} seconds")

