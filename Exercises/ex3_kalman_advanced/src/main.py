import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

MS2SEC = 100000
DATA_PATH = '../data/obj_pose-laser-radar-synthetic-input.txt'
LIDAR_HEADER = 'sensor_type,x_measured,y_measured,timestamp,' \
               'x_groundtruth,y_groundtruth,vx_groundtruth,' \
               'vy_groundtruth,yaw_groundtruth,yawrate_groundtruth'.split(',')


def kalmanFilter(state: np.ndarray, P: np.ndarray, R: np.ndarray,
                 F: np.ndarray, H: np.ndarray, measurment: np.ndarray) -> (np.ndarray, np.ndarray):
    # Stage I: Predict
    x = F.dot(state)
    P = F.dot(P.dot(F.T))

    # Stage II: Update
    K = P.dot(H.T).dot(
        np.linalg.inv(
            H.dot(P.dot(H.T)) + R
        )
    )

    state_out = x + K.dot(measurment - H.dot(x))
    P_out = (1 - K.dot(H)).dot(P)

    return state_out, P_out


def RMSE(a, b):
    return np.sqrt(np.square(a - b).sum())


def main():
    init_state = np.array([5,  # Pos X
                           5,  # Pos Y
                           0,  # Vel X
                           0])  # Vel y
    init_p = np.eye(4) * 12
    F = lambda dt: np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    R = np.diag([0.0255, 0.0255])
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ])  # TODO: figure out the matrix

    # Read data
    # Reading the data file, while skipping the Radar data lines and setting the headers
    data = pd.read_csv(DATA_PATH, sep='\t', header=None, comment='R', names=LIDAR_HEADER)
    print(data.head())

    new_p = init_p.copy()
    state = init_state.copy()

    last_dt = data.iloc[0]['timestamp'] - MS2SEC
    for data_line in data.iterrows():
        meas = np.array([
            data_line[1]['x_measured'],
            data_line[1]['y_measured']])
        gt = np.array([
            data_line[1]['x_groundtruth'],
            data_line[1]['y_groundtruth']])

        dt = data_line[1]['timestamp'] - last_dt
        dt /= MS2SEC
        last_dt = data_line[1]['timestamp']

        state, new_p = kalmanFilter(state, new_p, R, F(dt), H, meas)

        plt.plot(gt[0], gt[1], 'g*')
        plt.plot(meas[0], meas[1], 'ro')
        plt.plot(state[0], state[1], 'b*')
        plt.legend(['GT', 'Meas', 'KF'], loc=2)
        plt.title("RMSE {:.3f}".format(RMSE(state[:2], gt[:2])))
        plt.pause(.1)

    plt.show()


if __name__ == '__main__':
    main()
