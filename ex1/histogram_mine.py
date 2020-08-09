##
# A collection of functions to search in lists.
#
##
import sys

import numpy as np
import matplotlib.pyplot as plt
idx_c = 0

def normalize(arr):
    return arr / (arr.sum() + np.finfo('float').eps)


def sense(p, colors, measurement, sensor_right):
    h, w = p.shape[:2]
    hit = (measurement == colors)
    ret_q = p * (hit * sensor_right + (1 - hit) * (1 - sensor_right))

    # ret_q = np.zeros_like(p)
    # for x in range(ret_q.shape[1]):
    #     for y in range(ret_q.shape[0]):
    #         hit = (measurement == colors[y, x])
    #         ret_q[y, x] = p[y, x] * (hit * sensor_right + (1 - hit) * (1 - sensor_right))

    return normalize(ret_q)


def move(p, motion, p_move):
    h, w = p.shape[:2]
    Xs, Ys = np.meshgrid(range(w), range(h))
    Ys = (Ys - motion[0, 0]) % h
    Xs = (Xs - motion[0, 1]) % w
    ret_p = p_move * p[Ys, Xs] + (1 - p_move) * p
    return ret_p


def moveAndProc(movment):
    global colors_g, sensor_right_g, p_move_g, p_g, pos_g
    pos_g += np.array([[0, 0]]) if np.random.random() > p_move_g else movment
    pos_g[0, 0] = pos_g[0, 0] % colors_g.shape[0]
    pos_g[0, 1] = pos_g[0, 1] % colors_g.shape[1]
    f_meas = 'R' if colors_g[pos_g[0, 0], pos_g[0, 1]] == 'G' else 'G'
    measurement = f_meas if np.random.random() > sensor_right_g else colors_g[pos_g[0, 0], pos_g[0, 1]]

    p_g = move(p_g, movment, p_move_g)
    p_g = sense(p_g, colors_g, measurement, sensor_right_g)

    return p_g


def press(event):
    global fig, ax, p_g, target_g, best_guess,idx_c

    sys.stdout.flush()
    motion = np.zeros((1, 2), dtype=np.int)
    if event is not None:
        if event.key == 'up':
            motion = np.array([[-1, 0]])
        if event.key == 'down':
            motion = np.array([[1, 0]])
        if event.key == 'right':
            motion = np.array([[0, 1]])
        if event.key == 'left':
            motion = np.array([[0, -1]])

    motion = np.sign(target_g - best_guess)
    # max_idx = np.random.choice(np.argwhere(abs(motion) == max(abs(motion)))[:, 1])
    # motion[0, 1 - max_idx] = 0

    p_g = moveAndProc(motion)
    p_flat = p_g.flatten()
    p_flat.sort()
    ax.clear()
    ax.matshow(p_g, cmap='Blues')
    ax.plot(pos_g[0, 1], pos_g[0, 0], 'g*')
    ax.plot(target_g[0, 1], target_g[0, 0], 'y*')

    sam_n = 1
    best_guess = np.argwhere(p_g >= p_flat[-sam_n])[:sam_n]
    ax.plot(best_guess[0,1], best_guess[0,0], 'ro')
    plt.savefig('out' + "/file%02d.png" % idx_c)
    idx_c +=1
    fig.canvas.draw()


def histogram_localization(colors, sensor_right, p_move):
    global fig, ax, colors_g, sensor_right_g, p_move_g, p_g, pos_g, target_g, best_guess
    # initializes p to a uniform distribution over a grid of the same dimensions as colors
    vMap = np.zeros_like(colors)
    vMap[colors == 'G'] = 0
    vMap[colors == 'R'] = 1

    pos_g = np.random.randint(0, colors.shape[0], (1, 2))
    target_g = np.random.randint(0, colors.shape[0], (1, 2))
    p_move_g = p_move
    sensor_right_g = sensor_right

    colors_g = np.array(colors)
    p_g = np.ones_like(colors, dtype=np.float32) / colors.size
    best_guess = np.argwhere(p_g == p_g.max())[:1]
    # plt.imshow(vMap.astype(int))
    fig, ax = plt.subplots()
    fig.canvas.mpl_connect('key_press_event', press)
    ax.matshow(p_g, cmap='Blues')
    ax.plot(pos_g[0, 1], pos_g[0, 0], 'g*')
    ax.plot(target_g[0, 1], target_g[0, 0], 'y*')
    plt.ion()
    plt.show()
    while True:
        press(None)
        plt.pause(0.001)
        if np.all(target_g == best_guess):
            break

    print("Done")
    plt.ioff()
    plt.show()
    return p_g


def show(p):
    print(p)
