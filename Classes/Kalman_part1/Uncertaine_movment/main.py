# Main file of the Python 3 program.
import numpy as np


def update(mean1, var1, mean2, var2):
    var = (var1 + var2)
    mean = mean1 + mean2
    return mean, var


print(update(3, 4.5, 9, 1.3))
# answer is [12.0, 5.8]

print(update(3, 4.5, 9, 0))
# answer is [12.0, 4.5]

