import random
import copy

import numpy as np

simple_reward_map = {-6:-2,-5:-2,-4:-2 , -3:-2,-2: -1, -1: -1, 0: -3 , 1: 1, 2: 4, 3: -1}

gamma=.95
def get_tiles(robot, grid, values, policy_value):

    move = robot.dirs[policy_value]
    # Fool the robot and show a death tile as normal (dirty)
    data = {}

    to_check = tuple(np.array(robot.pos) + (np.array(move) * (1)))
    if to_check[0] < robot.grid.cells.shape[0] and to_check[1] < robot.grid.cells.shape[1] and to_check[
        0] >= 0 and to_check[1] >= 0:

        data[tuple(np.array(move))] = values[to_check]
            # print(data)
        

    return data



def policy_eval(grid, robot, values, policy):
    robot2 = copy.deepcopy(robot)

    values3 = copy.deepcopy(values)

    for iter in range(2):
        values2 = np.zeros_like(grid)

        for x in range(0, len(grid)):
            for y in range(0, len(grid[0])):

                robot2.pos = (x, y)

                if policy[x,y] == '-':
                    continue
       
                move =  robot.dirs[policy[x,y]]
                move_coord = tuple(np.array(robot2.pos) + (np.array(move)))
                value = values3[move_coord]

                # print((x,y), move_coord, policy[x,y],simple_reward_map[grid[move_coord]])
                if iter==0:
                    val = simple_reward_map[grid[move_coord]]
                else:
                    # Reward + 0.7*previous
                    val = gamma*value + simple_reward_map[grid[move_coord]]

                values2[x,y] = val
        
        if np.max(abs(values-values2)) < .01:
            print(f"{iter} iterations", np.max(values-values2))

            return values2

        values3 = copy.deepcopy(values2)

    return values3


def get_greedy_directions(values, robot):
    robot2 = copy.deepcopy(robot)
    directions = np.empty_like(values, dtype=str)

    for x in range(0, len(values)):
        for y in range(0, len(values[0])):
            robot2.pos = (x, y)

            possible_tiles = robot2.possible_tiles_after_move()
            possible_tiles = {move:possible_tiles[move] for move in possible_tiles if possible_tiles[move] > -1}

            values1 = [values[robot2.pos[0]+move[0], robot2.pos[1]+move[1]] for move in possible_tiles if possible_tiles[move] > -1]
            if len(values1) == 0:
                directions[x,y] = '-'
            else:

                best_move_index = np.argmax(values1)
                best_move = list(possible_tiles.keys())[best_move_index]
                new_orient = list(robot.dirs.keys())[list(robot.dirs.values()).index(best_move)]

                directions[x,y]=new_orient
    return directions

def policy_improv(policy, values, robot):

    greedy_directions  =get_greedy_directions(values, robot)

    stable_policy=True

    for x in range(0, len(policy)):
        for y in range(0, len(policy[0])):
            if policy[x,y] == '-':
                continue

            if greedy_directions[x,y] != policy[x,y]:
                policy[x,y] = greedy_directions[x,y]
                stable_policy=False
    
    return stable_policy


def gen_policy(robot, grid):
    robot2 = copy.deepcopy(robot)
    policy = np.empty_like(grid, dtype=str)

    for x in range(0, len(grid)):
        for y in range(0, len(grid[0])):
            robot2.pos = (x, y)

            directions = list(robot.dirs.keys())

            found_dir =False

           # Remove walls etc
            if grid[robot2.pos] < 0 and grid[robot2.pos] > -3:
                policy[x, y]='-'
                continue
                

            for dir in directions:
                move = robot.dirs[dir]
                to_check = tuple(np.array(robot2.pos) + (np.array(move)))

                if to_check[0] < robot.grid.cells.shape[0] and to_check[1] < robot.grid.cells.shape[1] and to_check[
                    0] >= 0 and to_check[1] >= 0:
                    
                    # Remove walls etc
                    if grid[to_check] < 0 and grid[to_check] > -3:
                        continue
                        
                    policy[x, y] = dir
                    found_dir = True

                    break

            if not found_dir:
                policy[x, y]='-'
            # print(data)
     
      
    return policy

def robot_epoch(robot):
    # Get the possible values (dirty/clean) of the tiles we can end up at after a move:
    print("Grid")
    print(np.vectorize(lambda x: simple_reward_map[x])(robot.grid.cells).T)

    # possible_tiles = robot.possible_tiles_after_move()
    # possible_tiles = {move:possible_tiles[move] for move in possible_tiles if possible_tiles[move] > -1}

    policy = gen_policy(robot, robot.grid.cells)
    print(policy.T)

    values = np.zeros_like(policy, dtype=float)

    for iter in range(5):
        print(f"Step {iter}") 
        # print(iter)
        values = policy_eval(robot.grid.cells, robot, values, policy)
        print(values.T)

        stable_policy = policy_improv(policy, values, robot)
        print(policy.T)
        if stable_policy:
            print(f"stable after {iter+1} steps")
            break
        print()

    new_orient = policy[robot.pos[0], robot.pos[1]]
    # Orient ourselves towards the dirty tile:
    while new_orient != robot.orientation:

        robot.rotate('r')
    # Move:
    robot.move()
