import numpy as np
import math


def get_poisson(lam, max_n):
    f = lambda n: lam ** n * math.exp(-lam) / math.factorial(n)
    p = [f(n) for n in range(max_n + 1)]
    s = sum(p[:-1])
    probs = []
    for i in range(max_n + 1):
        probs.append([0.0] * i + p[:-i-1] + [1.0 - s])
        s -= probs[-1][-2]
    return np.array(probs, dtype=np.float)


def get_cars_rented(rents_no_prob):
    for (y,x) in np.ndindex(rents_no_prob.shape):
        rents_no_prob[y][x] *= rents_no_prob.shape[0] + rents_no_prob.shape[1] - y - x - 2
    return rents_no_prob


def get_rewards(rents1, rents2, car_cost):
    reward = np.empty_like(rents1)
    for (y,x) in np.ndindex(rents1.shape):
        rents_no_prob = np.outer(rents1[y][:y + 1], rents2[x][:x + 1]) 
        reward[y][x] = np.sum(car_cost * get_cars_rented(rents_no_prob)) 
    return reward


def get_possible_actions(max_cars, max_move):
    shape = max_cars + 1, max_cars + 1
    possible_actions = np.empty(shape, dtype=object)
    for (y, x) in np.ndindex(shape):
        possible_actions[y][x] = np.arange(-min(x, max_cars - y, max_move), 
                min(y, max_cars - x, max_move) + 1, dtype=np.int)
    return possible_actions


def get_probs(rents1, returns1, rents2, returns2):
    return np.einsum('ij,kl->ikjl', rents1 @ returns1, rents2 @ returns2)


def get_policy(rewards, possible_actions, probs, discount_factor, max_cars, move_cost):
    next_utility = np.ones((max_cars + 1, max_cars + 1))
    utility = np.zeros((max_cars + 1, max_cars + 1))
    policy = np.zeros((max_cars + 1, max_cars + 1), dtype=np.int)
    while not np.array_equal(utility, next_utility):
        utility = np.copy(next_utility)
        for (y, x) in np.ndindex(utility.shape):
            m = 0
            for a in possible_actions[y][x]:
                r = rewards[y - a][x + a] - abs(a) * move_cost 
                u = r + discount_factor * np.sum(probs[y - a][x + a] * utility)
                if u > m:
                    policy[y][x] = a
                    next_utility[y][x] = u
                    m = u
    return policy


def print_policy(policy, max_cars):
    for i in range(max_cars + 1):
        for j in range(max_cars + 1):
            print("{:2d} ".format(policy[i][j]), end='')
        print()


if __name__ == "__main__":

    CAR_COST = 100

    MOVE_COST = 20
    
    RENTS1 = 3
    RENTS2 = 4
    RETURNS1 = 3
    RETURNS2 = 2
    
    MAX_CARS = 20
    MAX_MOVE = 5

    DISCOUNT_FACTOR = 0.9

    rents1 = np.flipud(np.fliplr(get_poisson(RENTS1, MAX_CARS)))
    rents2 = np.flipud(np.fliplr(get_poisson(RENTS2, MAX_CARS)))
    returns1 = get_poisson(RETURNS1, MAX_CARS)
    returns2 = get_poisson(RETURNS2, MAX_CARS)
    
    rewards = get_rewards(rents1, rents2, CAR_COST)

    possible_actions = get_possible_actions(MAX_CARS, MAX_MOVE)

    probs = get_probs(rents1, returns1, rents2, returns2)

    policy = get_policy(rewards, possible_actions, probs, DISCOUNT_FACTOR, MAX_CARS, MOVE_COST)
    print_policy(policy, MAX_CARS)
