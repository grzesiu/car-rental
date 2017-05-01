from abc import ABCMeta, abstractmethod
import numpy as np
import math
import copy

class PoissonMatrix(metaclass = ABCMeta):
    def __init__(self, vector):
        self.vector = vector
        self.matrix = []
        self.init_matrix()
        self.matrix = np.array(self.matrix)

    def init_matrix(self):
        for v in self.vector:
            self.matrix.append(v)

    @classmethod
    def left_shift(cls, lam, max_n):
        return cls(LeftShift(lam, max_n))

    @classmethod
    def right_shift(cls, lam, max_n):
        return cls(RightShift(lam, max_n))


class BottomUp(PoissonMatrix):
    def init_matrix(self):
        super().init_matrix()
        self.matrix = list(reversed(self.matrix))

class TopDown(PoissonMatrix):
    pass    

class PoissonVector(metaclass = ABCMeta):
    def __init__(self, lam, max_n):
        self.max_n = max_n
        poisson = lambda n: lam ** n * math.exp(-lam) / math.factorial(n)
        self.probs = [poisson(n) for n in range(max_n)]
        self.probs.append(1 - sum(self.probs))

    def __iter__(self):
        for _ in range(self.max_n + 1):
            yield copy.copy(self.probs)
            self.shift()

    @abstractmethod
    def shift(self):
        pass


class LeftShift(PoissonVector):
    def shift(self):
        self.probs[1] += self.probs[0]
        self.probs = self.probs[1:] + [0]


class RightShift(PoissonVector): 
    def shift(self):
        self.probs[-2] += self.probs[-1]
        self.probs = [0] + self.probs[:-1]

if __name__ == "__main__":
    np.set_printoptions(formatter={'float': '{: 0.2f}'.format}, linewidth=200)


    probs = BottomUp.right_shift(4, 20)
    print(probs.matrix)
    print(probs.matrix.shape)

    probs = TopDown.left_shift(4, 20)
    print(probs.matrix)
    print(probs.matrix.shape)
