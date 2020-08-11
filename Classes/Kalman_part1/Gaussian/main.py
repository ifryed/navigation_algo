##
# Main function of the Python program.
#
##

from finder import *
from scipy.stats import norm
import numpy as np
import math


def main():
    # we print a heading ComputeGausianand make it bigger using HTML formatting
    print(ComputeGausian(8, 2, 8))
    print(ComputeGausian(8, 10, 8))
    print(ComputeGausian(10, 1.2, 10))

    print("{:.6},{:.4}".format(*predict(5, 0, 15, 2)))
    print("{:.2f},{:.4f}".format(*predict(5, 1000, 15, 2)))

    print("{:.3}".format(GaussPDF(100, 15, 125)))
    print("{:.3}".format(GaussPDF(115, 20, 125)))


if __name__ == '__main__':
    main()
