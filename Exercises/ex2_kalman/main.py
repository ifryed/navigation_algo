import numpy as np
from numpy.linalg import inv

kFEET2METER = 0.3058

sen_pos_sig = 0.5 * kFEET2METER
sen_v_sig = 4
R_0 = np.array([
    [np.square(sen_pos_sig), sen_pos_sig * sen_v_sig],
    [sen_pos_sig * sen_v_sig, np.square(sen_pos_sig)],
])
dt = 1
P_0 = np.array([
    [np.square(sen_pos_sig), sen_pos_sig * sen_v_sig],
    [sen_pos_sig * sen_v_sig, np.square(sen_pos_sig)],
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
    :return:
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
        P = (1 - K.dot(H)).dot(P)

    return x, P


def main():
    initial_xy = np.array([[8.],  # Initial state (Position and Velocity)
                           [5.]])
    measurements = [[43 * kFEET2METER, 4],
                    [43 * kFEET2METER + 5, 6],
                    [43 * kFEET2METER + 11, 6],
                    ]

    print('Initial Status:\t{}'.format(initial_xy.T[0]))
    print('Measurments:\t{}'.format('\n\t\t\t\t'.join(['[{:.3f},{}]'.format(*x) for x in measurements])))

    x2, P2 = kalman_filter(initial_xy, P_0, H_0, F_0, R_0, measurements)
    print('x= ', x2)
    print('P= ', P2)


if __name__ == '__main__':
    main()
