# An implementation of dynamic programming using policy iteration
import numpy as np

simple_reward_map = {-2: 0, -1: 0, 0: 0, 1: 1, 2: 2, 3: -1}
discount_factor = 0.7
threshold_theta = 0.1


def robot_epoch(robot):
    vision = robot.possible_tiles_after_move()
    print(vision)

    policy = initialize_policy(vision)
    value_func = {move: 0 for move in vision}
    f_policy = improve_policy(policy, vision, value_func)
    chosen_move = f_policy[(0, 0)]
    # Rotation code stolen from greedy robot
    new_orient = list(robot.dirs.keys())[list(robot.dirs.values()).index(chosen_move)]
    while new_orient != robot.orientation:
        robot.rotate('r')
    robot.move()


def initialize_policy(vision):
    start_state = (0, 0)
    start_moves = get_neighbouring_moves(vision, start_state)
    print(start_moves)
    policy = {start_state: None}
    for s in vision:
        s_primes = get_neighbouring_moves(vision, s)
        if len(s_primes) > 0:
            policy[s] = None
    return policy


def get_neighbouring_moves(vision, move):
    move_set = {}
    for p_move in vision:
        state = vision[p_move]
        if abs(p_move[0] - move[0]) + abs(p_move[1] - move[1]) == 1:
            move_set[p_move] = state
    return move_set


def evaluate_policy(vision, policy, value_func):
    stop_criterion = False

    global simple_reward_map
    global discount_factor
    global threshold_theta

    while not stop_criterion:
        delta = 0
        for move in vision:
            move_value = value_func[move]
            next_possible_moves = get_neighbouring_moves(vision, move)

            if len(next_possible_moves) > 0:
                value_func[move] = 0
            for s_prime in next_possible_moves:
                value_func[move] += (policy[move] == s_prime) * (
                        simple_reward_map[vision[s_prime]] + discount_factor * value_func[s_prime])
            delta = max(delta, abs(move_value - value_func[move]))
        if delta < threshold_theta:
            stop_criterion = True
    return value_func


def improve_policy(policy, vision, value_func):
    policy_stable = False

    global simple_reward_map
    global discount_factor

    state_space = vision.copy()
    state_space[(0, 0)] = 0

    print(policy)

    while not policy_stable:
        for move in state_space:
            possible_actions = get_neighbouring_moves(vision, move)

            if len(possible_actions) < 1:
                continue
            taken_action_under_pi = policy[move]
            best_action = {move: 0 for move in possible_actions}

            for s_prime in possible_actions:
                best_action[s_prime] = (simple_reward_map[vision[s_prime]] + discount_factor * value_func[s_prime])
            best_move = max(best_action, key=best_action.get)
            if taken_action_under_pi == best_move:
                policy_stable = True
            else:
                print("Updated policy")
                policy[move] = best_move
                print(policy)
                value_func = evaluate_policy(vision, policy, value_func)

    print("Final policy is: ", policy)
    return policy
