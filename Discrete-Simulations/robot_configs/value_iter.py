######
# Value iteratation agent:
#  Makes a move based on policy generated by the value iteration algorithm
#  Main loop can be found in robot_epoch
######

import copy
import numpy as np

# Reward map that provides the scores associated with tiles. -3<=current position, -2&-1=walls&obstacles, 0=clean, 1=dirty, 2=goal
simple_reward_map = {-6:-2,-5:-2,-4:-2 ,-3:-2,-2: -9, -1: -9, 0: -2 , 1: 2, 2: 4, 3: -1}

# Discount factor
gamma=.9

def get_actions(robot, grid, values):
    """
    Returns the possible tiles (actions) for a robot position. Skips the current tile, unless it 
    is the only available tile (see report).

    @robot: robot that is on the requested position
    @grid: original grid with tile value
    @values: values calculated in the previous iterations
    
    """
    own=None
    data = {}
    
    moves = list(robot.dirs.values())
    
    # Check for all moves if they are possible tiles/actions
    for move in moves:
        move_coord = tuple(np.array(robot.pos) + (np.array(move)))

        # Filter out out of bounds tiles
        if move_coord[0] < robot.grid.cells.shape[0] and move_coord[1] < robot.grid.cells.shape[1] and move_coord[0] >= 0 and move_coord[1] >= 0:
            
            # Skip walls, obstacles and current position. Save latter for if no other tiles are found.
            if grid[move_coord] < 0:
                if grid[move_coord] <= -3:
                    own= (move, move_coord)
                
                continue
                
            data[tuple(np.array(move))] = values[move_coord]
     
    # If no other tiles are viable, add current position 
    if own and len(data) == 0:
        move, move_coord = own
        data[tuple(np.array(move))] = values[move_coord]


    return data



def policy_eval(grid, robot, max_iter=1000):
    """
    Evaluates the value for each state (i.e. tile) in the grid by the reward and previous values of
    all possible actions. Continues until threshold is reached.

    @grid: original grid with tile value
    @robot: original robot that is on its actual position
    @max_iter: amount of iterations before stopping if threshold is not triggered
    
    """

    robot2 = copy.deepcopy(robot)

    # Inititial values are 0
    values = np.zeros_like(robot.grid.cells, dtype=float)


    for iter in range(max_iter):
        # keeps new values of this iteration
        values2 = np.zeros_like(grid)

        # Loops over entire grid
        for x in range(0, len(grid)):
            for y in range(0, len(grid[0])):
                # Move robot to new position
                robot2.pos = (x, y)
                
                # Get all possible actions in this position
                possible_actions = get_actions(robot2, grid, values)

                # Calculate for value for all actions: gamma (discount) * previous_value(action) + reward(action)
                dir_values = [gamma*values[tuple(np.array(robot2.pos) + (np.array(move)))] + simple_reward_map[grid[x+move[0], y+move[1]]] for move in possible_actions]
                
                # Take average of all actions
                values2[x,y] = sum(dir_values)/len(dir_values) if len(dir_values) > 0 else 0


        # If difference between iterations if below threshold return current values
        if np.max(abs(values-values2)) < .0001:
            print(f"{iter} iterations", np.max(abs(values-values2)))

            return values2

        values = copy.deepcopy(values2)

    return values

def get_greedy_directions(values, robot):
    """
    Evaluates and returns the best direction (i.e. action) to take based greedily choosing 
    the one with the highest value calulated in the policy evaluation

    @robot: original robot that is on its actual position
    @values: values calculated in policy_iteration
    
    """
    robot2 = copy.deepcopy(robot)
    directions = np.empty_like(values, dtype=str)

    for x in range(0, len(values)):
        for y in range(0, len(values[0])):
            # Get possible tiles in for robot in position x,y
            robot2.pos = (x, y)
            possible_tiles = robot2.possible_tiles_after_move()

            # Filter out walls and obstacles (!=-9)
            possible_tiles = {move:possible_tiles[move] for move in possible_tiles if simple_reward_map[robot.grid.cells[robot2.pos[0]+move[0], robot2.pos[1]+move[1]]] != -9}

            values2 = [simple_reward_map[robot.grid.cells[robot2.pos[0]+move[0], robot2.pos[1]+move[1]]] +gamma*values[robot2.pos[0]+move[0], robot2.pos[1]+move[1]] for move in possible_tiles]
            
            # If not possible tiles return '-' as direction else take the move with highest value
            if len(values2) == 0:
                directions[x,y] = '-'
            else:

                best_move_index = np.argmax(values2)
                best_move = list(possible_tiles.keys())[best_move_index]
                new_orient = list(robot.dirs.keys())[list(robot.dirs.values()).index(best_move)]

                directions[x,y]=new_orient

    return directions


def robot_epoch(robot):
    """
    Makes a move for the robot based on the policy generated
    in the policy iteration step

    @robot: original robot that is on its actual position
    
    """
    # Get vaues from policy evaluation step
    values = policy_eval(robot.grid.cells, robot)

    # Create policy from greedy moves
    policy = get_greedy_directions(values, robot)

    # Get direction (orientation) to move in from the policy 
    new_orient = policy[robot.pos[0], robot.pos[1]]

    # Orient ourselves towards the dirty tile:
    while new_orient != robot.orientation:
        robot.rotate('r')
    # Move:
    robot.move()
