# Imports
import time
from continuous_environment import Grid, Robot, RobotAction, GymEnv

# Define environment and robots
grid = Grid(
    configFile="example-env.json",
    robots=[
        Robot(id=1, radius=0.5, color="blue", batteryLevel=100),
        Robot(id=2, radius=1, color="green", batteryLevel=100)
    ],
    startingPos=[
        [1, 1],
        [2, 2]
    ]
)

# Parameters
MAX_ITER = 100
DRAW = True
SAVE = True
GYM = True

gym_env = GymEnv(configFile="example-env.json",
                 robots=[
                     Robot(id=1, radius=0.5, color="blue", batteryLevel=100),
                     Robot(id=2, radius=1, color="green", batteryLevel=100)
                 ],
                 startingPos=[
                     [1, 1],
                     [2, 2]
                 ],
                 save=True)

if GYM:

    while True:
        # Take a random action
        actions = gym_env.action_space.sample()
        action_per_robot = [RobotAction(action) for action in actions]
        obs, reward, done, info = gym_env.step(action_per_robot)
        print(done)
        # Render the game
        gym_env.render()

        if done:
            break

else:
    # Main loop
    start_time = time.time()  # Measure time
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
    end_time = time.time()  # Measure time

    print(f"Time taken: {end_time - start_time} seconds\n")
    print()
    for robot in grid.robots:
        print(f"Original dirty area: {grid.totalGoalArea}")
        print(f"Robot {robot.id} cleaned area: {robot.areaCleaned}")
