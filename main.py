import time
from continuous_environment import Grid, Robot

grid = Grid(
    configFile="example-env.json", 
    robots=[
        Robot(id=1, radius=0.5, color="blue")
    ],
    startingPos=[
        [1, 1]
    ]
)


# Main loop
start_time = time.time()
while grid.moves < 100:
    if all([not robot.alive for robot in grid.robots]):
        print(f"{grid.moves} moves completed before death")
        break

    print(grid.moves)
    image = grid.plot_grid(resolution=512, draw=False, save=True)
    for robot in grid.robots:
        if robot.alive:
            robot.move(None, 1)

end_time = time.time()

print(f"Time taken: {end_time - start_time} seconds")

