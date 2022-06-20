# Imports
import time
from continuous_environment import Grid, Robot

# Define environment and robots
grid = Grid(
    configFile="example-env-full.json", 
    robots=[
        Robot(id=1, radius=0.15, color="blue", batteryLevel=100)
    ],
    startingPos=[
        [9.8, 5.3]
    ]
)

# Parameters
MAX_ITER = 100
DRAW = False
SAVE = True

image = grid.plot_grid(resolution=650, draw=DRAW, save=SAVE)
quit()


# Main loop
start_time = time.time() # Measure time
while max(grid.moves.values()) < MAX_ITER:
    print()
    # Check if environment is still active
    if all([not robot.alive for robot in grid.robots]):
        print(f"{max(grid.moves.values())} moves completed before death")
        break

    # Get environment image for robot
    image = grid.plot_grid(resolution=256, draw=DRAW, save=SAVE)

    # Move robots 
    for robot in grid.robots:
        print(f"Robot {robot.id} battery: {robot.batteryLevel}")
        if robot.alive:
            robot.move(None, 1)
    print(grid.moves)

# Final state
image = grid.plot_grid(resolution=256, draw=DRAW, save=SAVE)
end_time = time.time() # Measure time

print(f"Time taken: {end_time - start_time} seconds\n")
print()
for robot in grid.robots:
    print(f"Original dirty area: {grid.totalGoalArea}")
    print(f"Robot {robot.id} cleaned area: {robot.areaCleaned}")

