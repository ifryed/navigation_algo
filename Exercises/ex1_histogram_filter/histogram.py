##
# A collection of functions to search in lists.
#
##

import numpy as np


def normalize(arr):
    return arr / (arr.sum() + np.finfo('float').eps)


def sense(p, colors, measurement, sensor_right):
    hit = (measurement == colors)
    ret_q = p * (hit * sensor_right + (1 - hit) * (1 - sensor_right))

    return normalize(ret_q)


def move(p, motion, p_move):
    h, w = p.shape[:2]
    Xs, Ys = np.meshgrid(range(w), range(h))
    Ys = (Ys - motion[0]) % h
    Xs = (Xs - motion[1]) % w
    ret_p = p_move * p[Ys, Xs] + (1 - p_move) * p
    return ret_p


def histogram_localization(colors, measurements, motions, sensor_right, p_move):
    # initializes p to a uniform distribution over a grid of the same dimensions as colors
    colors = np.array(colors)
    p = np.ones_like(colors, dtype=np.float32) / colors.size

    for i in range(len(measurements)):
        p = move(p, np.array(motions[i]), p_move)
        p = sense(p, colors, measurements[i], sensor_right)

    return p


def show(p):
    rows = ['[' + ','.join(map(lambda x: '{0:.5f}'.format(x), r)) + ']' for r in p]
    print('[' + ',\n '.join(rows) + ']')
