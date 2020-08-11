##
# Main function of the Python program.
#
##
import numpy as np
from math import *
from numpy.linalg import inv

dt = 0.01
P = np.eye(4)
F = np.array([
    [1, 0, dt, 0],
    [0, 1, 0, dt],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
])
B = np.array([
    [dt ** 2 / 2],
    [dt ** 2 / 2],
    [dt],
    [dt]
])
H = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
])
R = np.zeros(2)
I = np.eye(4)
Q = B.dot(B.T)


def filter(x, P, u, measurements):
    for n in range(len(measurements)):
        # prediction
        x = F.dot(x) + B * (u)
        P = F.dot(P.dot(F.T)) + Q * 0

        # measurement update
        Z = measurements[n]
        print(Z, x.flatten())
        # y =
        # S =
        K = P.dot(H.T.dot(inv(H.dot(P.dot(H.T)) - R)))
        x = x + K.dot(Z - H.dot(x))
        P = (1 - K.dot(H)).dot(P)

    print('x= ', x)
    print('P= ', P)
    return P


def main():
    # we print a heading and make it bigger using HTML formatting
    initial_xy = [2., 10.]
    x = np.array([[initial_xy[0]], [initial_xy[1]], [0.], [0.]])  # initial state (location and velocity)
    measurements = [[5., 10.], [6., 8.], [7., 6.], [8., 4.], [9., 2.], [10., 0.]]
    u = np.array([[0.], [0.], [0.], [0.]])
    P2 = filter(x, P, u, measurements)
    print(P2[0][0])


if __name__ == '__main__':
    main()
