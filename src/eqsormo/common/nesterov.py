# Generator to generate Nesterov lambdas for gradient descent
# see https://blogs.princeton.edu/imabandit/2013/04/01/acceleratedgradientdescent/

from functools import lru_cache


@lru_cache(maxsize=5)
def lmbda(step):
    if step == 0:
        return 0
    else:
        return (1 + (1 + 4 * lmbda(step - 1) ** 2) ** 0.5) / 2


@lru_cache(maxsize=5)
def gamma(step):
    return (1 - lmbda(step)) / lmbda(step + 1)
