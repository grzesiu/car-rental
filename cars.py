import numpy as np
import math

REWARD = 100
PENALTY = 20
RENTAL1 = (3, 3)
RENTAL2 = (4, 2)
MAX_CARS = 20
MAX_MOVED = 5
DISCOUNT_FACTOR = 0.9
N_E = 5

def poisson(lam, max_n):
    f = lambda n: lam ** n * math.exp(-lam) / math.factorial(n)
    probs = np.vectorize(f)(np.arange(21))
    probs[-1] = 1 - np.sum(probs[:-1])
    return probs


def rewards():
    pass

if __name__ == "__main__":
    print(poisson(5, 20))
