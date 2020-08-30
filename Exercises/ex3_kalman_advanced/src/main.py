import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from kalmanLib import ExtendedKalmanFilter
import time

DATA_PATH = '../data/obj_pose-laser-radar-synthetic-input.txt'
LIDAR_HEADERS = 'sensor_type,x_measured,y_measured,timestamp,' \
                'x_groundtruth,y_groundtruth,vx_groundtruth,' \
                'vy_groundtruth,yaw_groundtruth,yawrate_groundtruth'.split(',')


def RMSE(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(np.square(a - b)).flatten()


def updatePlot(gt: np.ndarray, meas: np.ndarray, state: np.ndarray, rmse: np.ndarray, disp_on=True) -> None:
    # Plot
    plt.plot(gt[0], gt[1], 'g*')
    plt.plot(meas[0], meas[1], 'ro')
    plt.plot(state[0], state[1], 'b*')
    plt.legend(['GT', 'Meas', 'KF'], loc=2)
    plt.title("RMSE x={:.3},y={:.3},vx={:.3},vy={:.3}".format(*rmse))
    if disp_on:
        plt.pause(.01)


def extractDataLine(data_line: pd.DataFrame) -> (np.ndarray, np.ndarray, float):
    meas = np.array([[
        data_line[1]['x_measured'],
        data_line[1]['y_measured']]]).T
    gt = np.array([[
        data_line[1]['x_groundtruth'],
        data_line[1]['y_groundtruth'],
        data_line[1]['vx_groundtruth'],
        data_line[1]['vy_groundtruth'],
    ]]).T

    time_stamp = data_line[1]['timestamp']

    return meas, gt, time_stamp


def main():
    init_state = np.array([[5,  # Pos X
                            5,  # Pos Y
                            0,  # Vel X
                            0,  # Vel y
                            ]]).T
    init_p = np.diag([12] * len(init_state))
    R = np.diag([0.0255, 0.0255])
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    # Read data
    # Reading the data file, while skipping the Radar data lines and setting the headers
    data = pd.read_csv(DATA_PATH, sep='\t', header=None, comment='R', names=LIDAR_HEADERS)
    print("Data sample\n", data.head())

    ekf = ExtendedKalmanFilter(init_state, init_p, R, H)

    last_dt = data.iloc[0]['timestamp']

    os.makedirs('../out', exist_ok=True)
    logger = open('../out/log_{}.txt'.format(np.floor(time.time()).astype(int)), 'w')

    est_log = []
    gt_log = []
    for i, data_line in enumerate(data.iloc[1:].iterrows()):
        meas, gt, time_stamp = extractDataLine(data_line)

        dt = (time_stamp - last_dt) / 1000000
        last_dt = time_stamp

        ekf.predict(dt)
        state, new_p = ekf.update(meas)

        rmse = RMSE(state, gt)
        est_log.append(state)
        gt_log.append(gt)
        logger.write('{}:{}\n'.format(i, rmse))

        if i % 5 == 0:
            # Save logs
            logger.flush()

        updatePlot(gt, meas, state, rmse, True)

    logger.close()
    error_log = np.array(est_log) - np.array(gt_log)
    f_rmse = np.sqrt(np.square(error_log).mean(0))
    print("RMSE: {}".format(f_rmse))
    plt.title("RMSE x={:.3},y={:.3},vx={:.3},vy={:.3}".format(*f_rmse.flatten()))
    plt.show()


if __name__ == '__main__':
    main()
