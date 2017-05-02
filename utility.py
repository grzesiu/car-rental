import poisson

def rentals_probs(lambdas1, lambdas2, max_cars):
    rental1 = rental_probs(lambdas1, max_cars)
#    print(rental1)
    rental2 = rental_probs(lambdas2, max_cars)
    return rental1 * rental2


def rental_probs(lambdas, max_cars):
    rents = poisson.BottomUp.right_shift(lambdas[0], max_cars)
#    print(rents.matrix)
    returns = poisson.TopDown.left_shift(lambdas[1], max_cars)
    return rents.matrix @ returns.matrix
