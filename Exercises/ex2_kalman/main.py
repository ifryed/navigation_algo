import numpy as np
from numpy.linalg import inv

kFEET2METER = 0.3048

sen_pos_sig = 0.5 * kFEET2METER
sen_v_sig = 4

init_pos_sig = 2
init_v_sig = 1.2 * kFEET2METER

R_0 = np.array([
    [np.square(sen_pos_sig), 0],
    [0, np.square(sen_pos_sig)],
])
dt = 1
P_0 = np.array([
    [np.square(init_pos_sig), 0],
    [0, np.square(init_v_sig)],
])
F_0 = np.array([
    [1, dt],
    [0, 1]
])
H_0 = np.array([
    [1, 0],
    [0, 1],
])


def kalman_filter(x: np.ndarray, P: np.ndarray, H: np.ndarray, F: np.ndarray, R: np.ndarray, measurements: list) -> (
        np.ndarray, np.ndarray):
    """
    Performs Kalman Filter estimation
    :param x: Initial status (position,velocity)
    :param P: The a posteriori estimate covariance
    :param H: The observation model
    :param F: The state-transition model
    :param R: The covariance of the observation noise
    :param measurements: The sensor readings at each :math:'\delta t' interval (position reading,velocity reading)
    :return: The state vector (location, speed) and the
    """

    for n in range(len(measurements)):
        # Stage I: Prediction
        x = F.dot(x)
        P = F.dot(P.dot(F.T))

        # Stage II: Measurement reading & Update
        Z = measurements[n]
        Z = np.array(Z).reshape(-1, 1)

        K = P.dot(H.T).dot(np.linalg.inv(H.dot(P.dot(H.T)) + R))
        x = x + K.dot(Z - H.dot(x))
        P = (np.eye(len(x)) - K.dot(H)).dot(P)

    return x, P


def main():
    initial_xy = np.array([[8.],  # Initial state (Position and Velocity)
                           [5.]])
    measurements = [[43 * kFEET2METER, 4],
                    # [43 * kFEET2METER + 5, 6],
                    # [43 * kFEET2METER + 11, 6],
                    ]

    print('Initial Status:\t{}'.format(initial_xy.T[0]))
    print('Measurments:\t{}'.format('\n\t\t\t\t'.join(['[{:.3f},{}]'.format(*x) for x in measurements])))

    x2, P2 = kalman_filter(initial_xy, P_0, H_0, F_0, R_0, measurements)
    print('x = ', x2)
    print('P = ', P2)


if __name__ == '__main__':
    main()
