import numpy as np


class ExtendedKalmanFilter:
    def __init__(self, init_state: np.ndarray, init_p: np.ndarray, R: np.ndarray, H: np.ndarray):
        self.state = init_state
        self.P = init_p
        self.R = R
        self.H = H
        self.I = np.eye(len(self.state))
        self.last_acc = np.zeros((2,1))

    def _createFMatrix(self, dt: float) -> np.ndarray:
        f_mat = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        return f_mat

    def _computeConvMatrix(self, dt: float, sig_x: float, sig_y: float) -> np.ndarray:
        A = np.array([
            [np.square(dt) / 2, 0],
            [0, np.square(dt) / 2],
            [dt, 0],
            [0, dt],
        ])

        P = np.array([
            [np.square(sig_x), 0],
            [0, np.square(sig_y)],
        ])

        Q = A.dot(P).dot(A.T)
        return Q

    def predict(self, dt: float) -> (np.ndarray, np.ndarray):
        f_mat = self._createFMatrix(dt)
        self.state = f_mat.dot(self.state)

        ax, ay = self.last_acc.flatten()
        Q = self._computeConvMatrix(dt, ax, ay)

        self.P = f_mat.dot(self.P).dot(f_mat.T) + Q

        return self.state, self.P

    def update(self, measurment: np.ndarray) -> (np.ndarray, np.ndarray):
        last_v = self.state[2:]
        K = self.P.dot(self.H.T).dot(
            np.linalg.pinv(
                self.H.dot(self.P.dot(self.H.T)) + self.R
            )
        )

        self.state = self.state + K.dot(measurment - self.H.dot(self.state))
        self.P = (self.I - K.dot(self.H)).dot(self.P)

        self.last_acc = self.state[2:] - last_v
        return self.state, self.P
