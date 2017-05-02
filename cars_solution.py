import numpy as np
import utility

class Simulation:

    REWARD = 100
    PENALTY = 20
    RENTAL1 = (3, 3)
    RENTAL2 = (4, 2)
    MAX_CARS = 21
    MAX_MOVED = 5
    DISCOUNT_FACTOR = 0.9

    def get_possible_actions(self, y, x):
        return np.arange(-min(x, self.MAX_CARS - y), min(y, self.MAX_CARS - x) + 1))

    def get_probs(self):
        pass

if __name__ == "__main__":
    s = Simulation()
    np.set_printoptions(formatter={'float': '{: 0.2f}'.format}, linewidth=200)
