import numpy as np
import utility
import poisson

class Simulation:

    REWARD = 100
    PENALTY = 20
    RENTAL1 = (3, 3)
    RENTAL2 = (4, 2)
    MAX_CARS = 20
    MAX_MOVED = 5
    DISCOUNT_FACTOR = 0.9


    def __init__(self):
        self.rental1 = self.rental_probs(self.RENTAL1)
        self.rental2 = self.rental_probs(self.RENTAL2)
        self.utilities = np.zeros((self.MAX_CARS + 1, self.MAX_CARS + 1))

        for (y, x) in np.ndindex(self.utilities.shape):
            self.utilities[y][x] = self.utility(y, x)

    def rental_probs(self, lambdas):
        rents = poisson.BottomUp.right_shift(lambdas[0], self.MAX_CARS)
        returns = poisson.TopDown.left_shift(lambdas[1], self.MAX_CARS)
        return rents.matrix @ returns.matrix

    def utility(self, y, x):
        action_utilities = np.empty(self.MAX_CARS + 1)
        for (index), action in np.ndenumerate(self.possible_actions(y, x)):
            (ya, xa) = (y - action, x + action)
            s = 0
            for (yn, xn) in np.ndindex(self.utilities.shape):
                s += self.summand(action, ya, xa, yn, xn)
            action_utilities[index] = s
        return np.max(action_utilities)


    def summand(self, action, ya, xa, yn, xn):
        return self.rental1[ya][yn] * self.rental2[xa][xn] * \
               (self.PENALTY * abs(action) + self.DISCOUNT_FACTOR * self.utilities[yn][xn])


    def possible_actions(self, y, x):
        return np.arange(-min(x, self.MAX_CARS - y), min(y, self.MAX_CARS - x) + 1)


if __name__ == "__main__":
    s = Simulation()
    np.set_printoptions(formatter={'float': '{: 0.2f}'.format}, linewidth=200)
    print(s.utilities)
