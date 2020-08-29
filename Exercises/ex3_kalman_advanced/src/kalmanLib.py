import numpy as np
from typing import Callable


class ExtendedKalmanFilter:
    def __init__(self, init_state: np.ndarray, init_p: np.ndarray, R: np.ndarray, H: np.ndarray):
        self.state = init_state
        self.P = init_p
        self.R = R
        self.H = H
        self.I = np.eye(len(self.state))

    def _extractState(self) -> (float, float, float, float, float, float):
        x, y, vx, vy = self.state.flatten()
        r = np.sqrt(np.square(x) + np.square(y))
        h = np.arctan2(vy, vx)
        return x, y, vx, vy, r, h

    def _createFMatrix(self, dt: float) -> np.ndarray:
        f_mat = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        return f_mat

    def predict(self, dt: float):
        f_mat = self._createFMatrix(dt)
        self.state = f_mat.dot(self.state)
        self.P = f_mat.dot(self.P).dot(f_mat.T)

    def update(self, measurment: np.ndarray):
        K = self.P.dot(self.H.T).dot(
            np.linalg.pinv(
                self.H.dot(self.P.dot(self.H.T)) + self.R
            )
        )

        self.state = self.state + K.dot(measurment - self.H.dot(self.state))
        self.P = (self.I - K.dot(self.H)).dot(self.P)

        return self.state, self.P
