import numpy as np


def H(state_vector: np.ndarray) -> np.ndarray:
    """
    Transforms the state vector from cartesian space to polar space
    :param state_vector: The state vector in cartesian space
    :return: The state vector in polar space
    """
    px, py, vx, vy = state_vector
    rho = np.sqrt(np.square(px) + np.square(py))
    phi = np.arctan2(py, px)
    rho_dot = (px * vx + py * vy) / rho

    return np.array([rho, phi, rho_dot])


def main():
    px_0, py_0, vx_0, vy_0 = 10, 15, 2, 3
    x = np.array([[px_0, py_0, vx_0, vy_0]]).T

    print("X=\t\t\t\t{}".format(x.T))
    print("Sensor Space=\t{}".format(H(x).T))


if __name__ == '__main__':
    main()
