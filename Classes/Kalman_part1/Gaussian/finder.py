##
# A collection of functions to search in lists.
#
##

import numpy as np
import math

EPS = np.float64(.01)


# Returns the minimum element in a list of integers
def GaussPDF(mu, sigma, x):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp((-(x - mu) ** 2 / (2 * sigma ** 2)))


def ComputeGausian(mu, sigma, x):
    p1 = (1 / np.sqrt(2. * np.pi * sigma)).astype(np.float64)
    p2 = np.exp(-.5 * (x - mu) ** 2 / 2 * sigma).astype(np.float64)
    return p1 * p2


def predict(mu1, var1, mu2, var2):
    var1 = (var1 + EPS) if var1 == 0 else var1
    var2 = (var2 + EPS) if var2 == 0 else var2

    mu = (mu1 * var2 + mu2 * var1) / (var1 + var2)
    var = 1 / (1 / var1 + 1 / var2)
    return mu, var


def update(mu1, var1, mu2, var2):
    pass
