import numpy as np
import math

CAR_COST = 100
MOVE_COST = 20

RENTS_1 = 3
RENTS_2 = 4
RETURNS_1 = 3
RETURNS_2 = 2

MAX_CARS = 20
MAX_MOVED = 5
DISCOUNT_FACTOR = 0.9
N_E = 5

def poisson(lam, max_n):
    f = lambda n: lam ** n * math.exp(-lam) / math.factorial(n)
    p = [f(n) for n in range(max_n + 1)]
    s = sum(p[:-1])
    probs = []
    for i in range(max_n + 1):
        probs.append([0.0] * i + p[:-i-1] + [1.0 - s])
        s -= probs[-1][-2]
    return np.array(probs, dtype=np.float)

def reward(outer):
    for (y,x) in np.ndindex(outer.shape):
        outer[y][x] *= outer.shape[0] + outer.shape[1] - y - x - 2
    return outer

def rewards(rents1, rents2, car_cost):
    r = np.empty_like(rents1)
    for (y,x) in np.ndindex(rents1.shape):
        outer = np.outer(rents1[y][:y + 1], rents2[x][:x + 1]) 
        r[y][x] = np.sum(car_cost * reward(outer)) 
    return r

#def cars_rented(max_n):
#    cars_rented = np.empty((max_n + 1, max_n + 1))
#    for (y, x) in np.ndindex(cars_rented.shape):
#        cars_rented[y][x] = 2 * max_n - y - x
#    return cars_rented

if __name__ == "__main__":
    np.set_printoptions(formatter={'float': '{: 0.2f}'.format}, linewidth=200)

    rents1 = np.flipud(np.fliplr(poisson(RENTS_1, MAX_CARS)))
    rents2 = np.flipud(np.fliplr(poisson(RENTS_2, MAX_CARS)))
    returns1 = poisson(RETURNS_1, MAX_CARS)
    returns2 = poisson(RETURNS_2, MAX_CARS)
    
    test1 = np.array([[3,0,0],[1,2,0],[0,1,2]])
    test2 = np.array([[5,0,0],[3,2,0],[1,2,2]])
    print(rents1)
    print(rents2)
    print(rewards(rents1, rents2, CAR_COST))
