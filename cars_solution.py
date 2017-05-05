import numpy as np
import poisson

class Simulation:

    REWARD = 100
    PENALTY = 20
    RENTAL1 = (3, 3)
    RENTAL2 = (4, 2)
    MAX_CARS = 10
    MAX_MOVED = 2
    DISCOUNT_FACTOR = 0.9


    def __init__(self):
        self.rental1 = self.rental_probs(self.RENTAL1)
        self.rental2 = self.rental_probs(self.RENTAL2)

        self.shape = (self.MAX_CARS + 1, self.MAX_CARS + 1)

        self.utils = np.zeros(self.shape)
        self.init_policy()

    def rental_probs(self, lambdas):
        rents = poisson.BottomUp.right_shift(lambdas[0], self.MAX_CARS)
        returns = poisson.TopDown.left_shift(lambdas[1], self.MAX_CARS)
        return rents.matrix @ returns.matrix

    def init_policy(self):
        self.policy = np.empty(self.shape, dtype=np.int)
        for (y, x) in np.ndindex(self.shape):
            self.policy[y][x] = np.random.choice(self.possible_actions(y, x))

    def run(self, k):
        z = 0
        while(True):
            for i in range(k):
                self.step()
            next_policy = self.next_policy()

            print(z)
            print("old")
            print(self.policy)
            print("next")
            print(next_policy)
            z += 1
            if np.array_equal(next_policy, self.policy):
                break
            else:
                self.policy = next_policy

    def step(self):
        next_utils = np.zeros(self.shape)
        for (y, x) in np.ndindex(self.shape):
            for (yn, xn) in np.ndindex(self.utils.shape):
                next_utils += self.summand(self.policy[y][x], y, x, yn, xn)
        self.utils = next_utils

    def summand(self, action, y, x, yn, xn):
        return self.rental1[y - action][yn] * self.rental2[x + action][xn] * \
            (self.PENALTY * abs(action) + self.DISCOUNT_FACTOR * self.utils[yn][xn])

    def next_policy(self):
        policy = np.empty(self.shape, dtype=np.int)
        for (y, x) in np.ndindex(self.shape):
            f = lambda a: sum([self.summand_policy(a, y, x, yn, xn) for (yn, xn) in np.ndindex(self.utils.shape)])
            policy[y][x] = max(self.possible_actions(y, x), key=f)
        return policy

    def summand_policy(self, action, y, x, yn, xn):
        return self.rental1[y - action][yn] * self.rental2[x + action][xn] * self.utils[yn][xn]

    def possible_actions(self, y, x):
        return np.arange(-min(x, self.MAX_CARS - y), min(y, self.MAX_CARS - x) + 1, dtype=np.int)

if __name__ == "__main__":
    np.set_printoptions(formatter={'float': '{: 0.2f}'.format}, linewidth=200)
    s = Simulation()
    s.run(5)

#cars_solution.py:56: RuntimeWarning: overflow encountered in add
#  next_utils += self.summand(self.policy[y][x], y, x, yn, xn)
#cars_solution.py:61: RuntimeWarning: invalid value encountered in double_scalars
#  (self.PENALTY * abs(action) + self.DISCOUNT_FACTOR * self.utils[yn][xn])


