# An implementation of dynamic programming using policy iteration
import numpy as np

simple_reward_map = {-3: 0, -2: -1, -1: -1, 0: 0, 1: 1, 2: 2, 3: -1}
discount_factor = 0.3
threshold_theta = 0.1
epsilon = 0.1
solved = False
optimal_policy = None
optimal_value_func = None

def robot_epoch(robot):
    #   vision = robot.possible_tiles_after_move()
    global solved
    global optimal_policy
    global optimal_value_func

    robo_position = robot.pos

    if not solved:
        move_set = {(j, i): robot.grid.cells[(j, i)] for i in range(robot.grid.n_rows) for j in
                    range(robot.grid.n_cols)}
        print(move_set)
        vision = move_set
        global epsilon

        # Do not include walls in the moveset
        vision = {move: vision[move] for move in vision if 0 <= vision[move] <= 3}
        #  print(vision)

        policy = initialize_policy(vision, robot)
        print("Past init")
        optimal_value_func = {move: 0 for move in vision}
        optimal_policy, optimal_value_func = improve_policy(policy, vision, optimal_value_func, robot)
        print("Got back")
        #solved = True
    else:
        vision = {(j, i): robot.grid.cells[(j, i)] for i in range(robot.grid.n_rows) for j in
                    range(robot.grid.n_cols)}
        vision = {move: vision[move] for move in vision if 0 <= vision[move] < 3}
        optimal_policy = improve_policy(optimal_policy, vision, optimal_value_func, robot)

    print(optimal_policy)
    print(robo_position)
    chosen_move = optimal_policy[robo_position]
    print("Have chosen move boi ", chosen_move, "value func ", optimal_value_func)
    new_orient_dir = (chosen_move[0] - robo_position[0], chosen_move[1] - robo_position[1])
    print("Orienting to ", new_orient_dir)
    # Rotation code stolen from greedy robot
    new_orient = list(robot.dirs.keys())[list(robot.dirs.values()).index(new_orient_dir)]
    while new_orient != robot.orientation:
        robot.rotate('r')
    robot.move()


def initialize_policy(vision, robot):
    robo_position = robot.pos
    start_state = robo_position
    start_moves = get_neighbouring_moves(vision, start_state, robot)
 #   print(start_moves)
    policy = {start_state: None}
    for s in vision:
        s_primes = get_neighbouring_moves(vision, s, robot)
    #    print("Doing ", s)
        if len(s_primes) > 0:
            policy[s] = None
    return policy


def get_neighbouring_moves(vision, move, robot):  # Optimize
    move_set = {}
    for d in robot.dirs:
        #   print("Dir ", d, robot.grid.n_cols, robot.grid.n_rows)
        result_dir = tuple(np.array(move) + np.array(robot.dirs[d]))
        #   print(np.array(vision))
        #    print(result_dir)
        if robot.grid.n_cols > result_dir[0] >= 0 and robot.grid.n_rows > result_dir[1] >= 0 and result_dir in vision:
            state = vision[result_dir]
            move_set[result_dir] = state
    return move_set


def evaluate_policy(vision, policy, value_func, robot):
    stop_criterion = False

    global simple_reward_map
    global discount_factor
    global threshold_theta

    while not stop_criterion:
        delta = 0
        for move in vision:
            # $       print("Evaluating policy for ", move)
            move_value = value_func[move]
            next_possible_moves = get_neighbouring_moves(vision, move, robot)
            #     print("Neighbours are ", next_possible_moves)
            if len(next_possible_moves) > 0:
                value_func[move] = 0
            for s_prime in next_possible_moves:
                reward = 0
                if -3 <= int(vision[s_prime]) <= 3:
                    reward = simple_reward_map[int(vision[s_prime])] # Reward changes, after sweeping a square it has to be updated again
                #        print("Reward is ", reward)
                if policy[move] == s_prime:
                    value_func[move] += (
                            reward + discount_factor * value_func[s_prime])
            #      print("Got past loop")
            delta = max(delta, abs(move_value - value_func[move]))
        #       print("Got past delta")
        if delta < threshold_theta:
            stop_criterion = True
    return value_func


def improve_policy(policy, vision, value_func, robot):
    policy_stable = False

    global simple_reward_map
    global discount_factor

    robo_position = robot.pos

    state_space = vision.copy()
    state_space[robo_position] = 0

    print(vision)

    while not policy_stable:
        for move in state_space:
    #        print("Updating move ", move)
            possible_actions = get_neighbouring_moves(vision, move, robot)
    #        print("Neighbouring moves: ", possible_actions)
            if len(possible_actions) < 1:
                continue
            taken_action_under_pi = policy[move]
            best_action = {move: 0 for move in possible_actions}

            for s_prime in possible_actions:
                reward = 0
                if -3 <= int(vision[s_prime]) <= 3:
                    reward = simple_reward_map[int(vision[s_prime])]
                best_action[s_prime] = (reward + discount_factor * value_func[s_prime])
            best_move = max(best_action, key=best_action.get)
            if taken_action_under_pi == best_move:
                policy_stable = True
            else:
                #      print("Updated policy ", best_move)
                policy[move] = best_move
             #   print(policy)
                value_func = evaluate_policy(vision, policy, value_func, robot)

  #  print("Final policy is: ", policy)
    return policy, value_func
