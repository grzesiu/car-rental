import math

def get_poisson(lam, n):
    return lam ** n * math.exp(-lam) / math.factorial(n)

if __name__ == "__main__":
    print(get_poisson(10, 2))
