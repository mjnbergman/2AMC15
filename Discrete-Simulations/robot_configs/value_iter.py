import random
import copy

import numpy as np

simple_reward_map = {-6:-2,-5:-2,-4:-2 , -3:-2,-2: -9, -1: -9, 0: -2 , 1: 2, 2: 4, 3: -1}

gamma=.9

def get_tiles(robot, grid, values):
    moves = list(robot.dirs.values())
    # Fool the robot and show a death tile as normal (dirty)
    data = {}

    own=None
    for move in moves:
        to_check = tuple(np.array(robot.pos) + (np.array(move) * (1)))

        if to_check[0] < robot.grid.cells.shape[0] and to_check[1] < robot.grid.cells.shape[1] and to_check[
            0] >= 0 and to_check[1] >= 0:
            if grid[to_check] < 0:

                if grid[to_check] <= -3:
                    own= (move, to_check)
                continue
                
            data[tuple(np.array(move))] = values[to_check]
            # print(data)
     
    if own and len(data) == 0:
        move, to_check = own
        data[tuple(np.array(move))] = values[to_check]


    return data



def policy_eval(grid, robot):
    robot2 = copy.deepcopy(robot)

    values3 = np.zeros_like(robot.grid.cells, dtype=float)


    for iter in range(1000):
        values2 = np.zeros_like(grid)

        for x in range(0, len(grid)):
            for y in range(0, len(grid[0])):

                robot2.pos = (x, y)
                
                possible_tiles = get_tiles(robot2, grid, values3)

                dir_values = [gamma*values3[tuple(np.array(robot2.pos) + (np.array(move)))] + simple_reward_map[grid[x+move[0], y+move[1]]] for move in possible_tiles]
                values2[x,y] = sum(dir_values)/len(dir_values) if len(dir_values) > 0 else 0


        if np.max(abs(values3-values2)) < .0001:
            print(f"{iter} iterations", np.max(abs(values3-values2)))

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
            possible_tiles = {move:possible_tiles[move] for move in possible_tiles if simple_reward_map[robot.grid.cells[robot2.pos[0]+move[0], robot2.pos[1]+move[1]]] != -9}

            values1 = [simple_reward_map[robot.grid.cells[robot2.pos[0]+move[0], robot2.pos[1]+move[1]]] +gamma*values[robot2.pos[0]+move[0], robot2.pos[1]+move[1]] for move in possible_tiles]
            
            if len(values1) == 0:
                directions[x,y] = '-'
            else:

                best_move_index = np.argmax(values1)
                best_move = list(possible_tiles.keys())[best_move_index]
                new_orient = list(robot.dirs.keys())[list(robot.dirs.values()).index(best_move)]

                directions[x,y]=new_orient
    return directions




def robot_epoch(robot):
    # Get the possible values (dirty/clean) of the tiles we can end up at after a move:
    print("Grid")
    print(np.vectorize(lambda x: simple_reward_map[x])(robot.grid.cells).T)



    print(f"Step {iter}") 
    # print(iter)
    values = policy_eval(robot.grid.cells, robot)
    print(values.T)

    policy = get_greedy_directions(values, robot)

    new_orient = policy[robot.pos[0], robot.pos[1]]
    # Orient ourselves towards the dirty tile:
    while new_orient != robot.orientation:

        robot.rotate('r')
    # Move:
    robot.move()
