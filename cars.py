import numpy as np
import math

CAR_COST = 100
MOVE_COST = 20

RENTS1 = 3
RENTS2 = 4
RETURNS1 = 3
RETURNS2 = 2

MAX_CARS = 20
MAX_MOVE = 5
DISCOUNT_FACTOR = 0.9
N_E = 5


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
    r = np.empty_like(rents1)
    for (y,x) in np.ndindex(rents1.shape):
        rents_no_prob = np.outer(rents1[y][:y + 1], rents2[x][:x + 1]) 
        r[y][x] = np.sum(car_cost * get_cars_rented(rents_no_prob)) 
    return r


def get_possible_actions(max_cars, max_move):
    shape = max_cars + 1, max_cars + 1
    possible_actions = np.empty(shape, dtype=object)
    for (y, x) in np.ndindex(shape):
        possible_actions[y][x] = np.arange(-min(x, max_cars - y, max_move), 
                min(y, max_cars - x, max_move) + 1, dtype=np.int)
    return possible_actions


def get_init_policy(max_cars, possible_actions):
    policy = np.empty((max_cars + 1, max_cars + 1), dtype=np.int)
    for s in np.ndindex(policy.shape):
        policy[s] = np.random.choice(possible_actions[s])
    return policy
   

def get_probs(rents1, returns1, rents2, returns2):
    return np.einsum('ij,kl->ikjl', rents1 @ returns1, rents2 @ returns2)


def get_next_utility(rewards, policy, utility, probs, discount_factor, move_cost):
    next_utility = np.empty_like(utility, dtype=np.float)
    for (y, x) in np.ndindex(utility.shape):
        ya = y - policy[y][x]
        xa = x + policy[y][x]
        r = rewards[ya][xa] - abs(policy[y][x]) * move_cost
        next_utility[y][x] = r + discount_factor * np.sum(probs[ya][xa] * utility)
    return next_utility


def get_next_policy(possible_actions, probs, utility):
    next_policy = np.empty_like(utility, dtype=np.int)
    for (y, x) in np.ndindex(next_policy.shape):
        for actions in possible_actions[y][x]:
            m = 0
            for a in np.nditer(actions):
                ua = np.sum(probs[y - a][x + a] * utility)
                if ua > m:
                    m = ua
                    next_policy[y][x] = a
    return next_policy


def iterate_policy(rewards, possible_actions, probs, discount_factor, max_cars, move_cost, k):
    print(possible_actions)
    next_policy = get_init_policy(max_cars, possible_actions)
    policy = np.empty_like(next_policy, dtype=np.int)
    utility = np.zeros_like(rewards, dtype=np.int)

    while not np.array_equal(next_policy, policy):
        policy = next_policy
        for _ in range(k):
            utility = get_next_utility(rewards, policy, utility, probs, discount_factor, move_cost)

        next_policy = get_next_policy(possible_actions, probs, utility)
        print(utility)
        print(next_policy)
    
    
if __name__ == "__main__":
    np.set_printoptions(formatter={'float': '{: 0.2f}'.format}, linewidth=100000)

    rents1 = np.flipud(np.fliplr(get_poisson(RENTS1, MAX_CARS)))
    rents2 = np.flipud(np.fliplr(get_poisson(RENTS2, MAX_CARS)))
    returns1 = get_poisson(RETURNS1, MAX_CARS)
    returns2 = get_poisson(RETURNS2, MAX_CARS)
    
    rewards = get_rewards(rents1, rents2, CAR_COST)

    possible_actions = get_possible_actions(MAX_CARS, MAX_MOVE)

    probs = get_probs(rents1, returns1, rents2, returns2)

    iterate_policy(rewards, possible_actions, probs, DISCOUNT_FACTOR, MAX_CARS, MOVE_COST, 1)
