import math


def poisson(lam, n):
    return lam ** n * math.exp(-lam) / math.factorial(n)


def poisson_probs(lam, max_n):
    probs = [poisson(lam, n) for n in range(max_n)]
    probs.append(1 - sum(probs))
    return probs


if __name__ == "__main__":
    probs = poisson_probs(4, 20)
    print(probs)
    print(len(probs))
