import numpy as np
import math

CAR_COST = 100
MOVE_COST = 20
RENTAL1 = (3, 3)
RENTAL2 = (4, 2)
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


if __name__ == "__main__":
    np.set_printoptions(formatter={'float': '{: 0.2f}'.format}, linewidth=200)
    print(poisson(5, 20))

