import constants
import utils
from numpy import random

class Cell:
    def __init__(self):
        self.utility = 0
        self.action = 0

class Board:
    def __init__(self, size):
        self.size = size
        self.cells = [[Cell()] * size] * size


class Simulation:

    REWARD = 100
    PENALTY = 20
    RENTS_NO_1 = 3
    RENTS_NO_2 = 4
    RETURNS_NO_1 = 3
    RETURNS_NO_2 = 2
    MAX_MOVED = 5
    DISCOUNT_FACTOR = 0.9
    MAX_CARS = 4

    def __init__(self):
        pass    

    def get_possible_actions(self, y, x):
        return list(range(-min(x, self.MAX_CARS - y), min(y, self.MAX_CARS - x) + 1))
    

    def get_rent_probs():
        rent_probs_1 = poisson_probs(RENTS_NO_1, MAX_CARS)
        rent_probs_2 = poisson_probs(RENTS_NO_2, MAX_CARS)

def dummy():
    M = 20
    policy = random.randint(-5, 5, (M+1, M+1))
    for i in range(M+1):
        for j in range(M+1):
            print("{:2d} ".format(policy[i][j]), end='')
        print()



if __name__ == "__main__":
    s = Simulation()
    print(s.get_possible_actions(1, 3))


    
    
