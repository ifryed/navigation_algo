# --------------
# USER INSTRUCTIONS
#
# Now you will put everything together.
#
# First make sure that your sense and move functions
# work as expected for the test cases provided at the
# bottom of the previous two programming assignments.
# Once you are satisfied, copy your sense and move
# definitions into the robot class on this page, BUT
# now include noise.
#
# A good way to include noise in the sense step is to
# add Gaussian noise, centered at zero with variance
# of self.bearing_noise to each bearing. You can do this
# with the command random.gauss(0, self.bearing_noise)
#
# In the move step, you should make sure that your
# actual steering angle is chosen from a Gaussian
# distribution of steering angles. This distribution
# should be centered at the intended steering angle
# with variance of self.steering_noise.
#
# Feel free to use the included set_noise function.
#
# Please do not modify anything except where indicated
# below.

from math import *
import random
import numpy as np
import matplotlib.pyplot as plt

# --------
#
# some top level parameters
#

max_steering_angle = pi / 4.0  # You do not need to use this value, but keep in mind the limitations of a real car.
bearing_noise = 0.1  # Noise parameter: should be included in sense function.
steering_noise = 0.1  # Noise parameter: should be included in move function.
distance_noise = 5.0  # Noise parameter: should be included in move function.

tolerance_xy = 15.0  # Tolerance for localization in the x and y directions.
tolerance_orientation = 0.25  # Tolerance for orientation.

# --------
#
# the "world" has 4 landmarks.
# the robot's initial coordinates are somewhere in the square
# represented by the landmarks.
#
# NOTE: Landmark coordinates are given in (y, x) form and NOT
# in the traditional (x, y) format!

landmarks = np.array([
    [0.0, 100.0],
    [0.0, 0.0],
    [100.0, 0.0],
    [100.0, 100.0],
])  # position of 4 landmarks in (y, x) format.
world_size = 100.0  # world is NOT cyclic. Robot is allowed to travel "out of bounds"


# ------------------------------------------------
#
# this is the robot class
#

class robot:

    # --------
    # init:
    #    creates robot and initializes location/orientation
    #

    def __init__(self, length=20.0):
        self.x = random.random() * world_size  # initial x position
        self.y = random.random() * world_size  # initial y position
        self.orientation = random.random() * 2.0 * pi  # initial orientation
        self.length = length  # length of robot
        self.bearing_noise = 0.0  # initialize bearing noise to zero
        self.steering_noise = 0.0  # initialize steering noise to zero
        self.distance_noise = 0.0  # initialize distance noise to zero

    # --------
    # set:
    #    sets a robot coordinate
    #

    def set(self, new_x, new_y, new_orientation):

        if new_orientation < 0 or new_orientation >= 2 * pi:
            raise ValueError('Orientation must be in [0..2pi]')
        self.x = float(new_x)
        self.y = float(new_y)
        self.orientation = float(new_orientation)

    # --------
    # set_noise:
    #    sets the noise parameters
    #
    def set_noise(self, new_b_noise, new_s_noise, new_d_noise):
        # makes it possible to change the noise parameters
        # this is often useful in particle filters
        self.bearing_noise = float(new_b_noise)
        self.steering_noise = float(new_s_noise)
        self.distance_noise = float(new_d_noise)

    # --------
    # measurement_prob
    #    computes the probability of a measurement
    #

    def measurement_prob(self, measurements):

        # calculate the correct measurement
        predicted_measurements = self.sense(0)  # Our sense function took 0 as an argument to switch off noise.

        # compute errors
        error = 1.0
        for i in range(len(measurements)):
            error_bearing = abs(measurements[i] - predicted_measurements[i])
            error_bearing = (error_bearing + pi) % (2.0 * pi) - pi  # truncate

            # update Gaussian
            error *= (exp(- (error_bearing ** 2) / (self.bearing_noise ** 2) / 2.0) /
                      sqrt(2.0 * pi * (self.bearing_noise ** 2)))

        return error

    def __repr__(self):  # allows us to print robot attributes.
        return '[x=%.6s y=%.6s orient=%.6s]' % (str(self.x), str(self.y),
                                                str(self.orientation))

    ############# ONLY ADD/MODIFY CODE BELOW HERE ###################

    # --------
    # move:
    #

    # copy your code from the previous exercise
    # and modify it so that it simulates motion noise
    # according to the noise parameters
    #           self.steering_noise
    #           self.distance_noise
    def move(self, motion):  # Do not change the name of this function

        # ADD CODE HERE
        result = robot(self.length)
        result.x = self.x
        result.y = self.y
        result.bearing_noise = self.bearing_noise
        result.steering_noise = self.steering_noise
        result.distance_noise = self.distance_noise

        angel, d = motion
        steering2 = random.gauss(angel, result.steering_noise)
        distance2 = random.gauss(d, result.distance_noise)

        beta = distance2 * tan(steering2) / result.length

        if abs(beta) < 0.001:
            result.orientation = (beta + self.orientation) % (2.0 * pi)
            result.x += distance2 * cos(result.orientation)
            result.y += distance2 * sin(result.orientation)
        else:
            R = distance2 / beta
            cx = self.x - (sin(self.orientation) * R)
            cy = self.y + (cos(self.orientation) * R)
            result.orientation = (beta + self.orientation) % (2.0 * pi)
            result.x = cx + (sin(result.orientation) * R)
            result.y = cy - (cos(result.orientation) * R)

        return result

    # --------
    # sense:
    #

    # copy your code from the previous exercise
    # and modify it so that it simulates bearing noise
    # according to
    #           self.bearing_noise
    def sense(self, hasNoise):  # do not change the name of this function
        Z = []

        for i in range(4):
            y, x = landmarks[i]
            dx, dy = (x - self.x, y - self.y)
            teta = hasNoise * self.bearing_noise + atan2(dy, dx)
            s = (teta - self.orientation) % (2 * pi)
            Z.append(s)
        return Z  # Leave this line here. Return vector Z of 4 bearings.

    ############## ONLY ADD/MODIFY CODE ABOVE HERE ####################


# --------
#
# extract position from a particle set
#

def get_position(p):
    x = 0.0
    y = 0.0
    orientation = 0.0
    for i in range(len(p)):
        x += p[i].x
        y += p[i].y
        # orientation is tricky because it is cyclic. By normalizing
        # around the first particle we are somewhat more robust to
        # the 0=2pi problem
        orientation += (((p[i].orientation - p[0].orientation + pi) % (2.0 * pi))
                        + p[0].orientation - pi)
    return [x / len(p), y / len(p), orientation / len(p)]


# --------
#
# The following code generates the measurements vector
# You can use it to develop your solution.
#


def generate_ground_truth(motions):
    myrobot = robot()
    myrobot.set_noise(bearing_noise, steering_noise, distance_noise)

    Z = []
    T = len(motions)
    pos = []
    for t in range(T):
        myrobot = myrobot.move(motions[t])
        Z.append(myrobot.sense(False))
        pos.append((myrobot.x, myrobot.y))
    # print 'Robot:    ', myrobot
    return [myrobot, Z, np.array(pos)]


def generate_ground_truth_loc(motions):
    myrobot = robot()
    myrobot.set_noise(bearing_noise, steering_noise, distance_noise)

    Z = []
    T = len(motions)

    for t in range(T):
        myrobot = myrobot.move(motions[t])
        Z.append((myrobot.x, myrobot.y))
    # print 'Robot:    ', myrobot
    return [myrobot, np.array(Z)]


# --------
#
# The following code prints the measurements associated
# with generate_ground_truth
#

def print_measurements(Z):
    T = len(Z)

    print
    'measurements = [[%.8s, %.8s, %.8s, %.8s],' % \
    (str(Z[0][0]), str(Z[0][1]), str(Z[0][2]), str(Z[0][3]))
    for t in range(1, T - 1):
        print
        '                [%.8s, %.8s, %.8s, %.8s],' % \
        (str(Z[t][0]), str(Z[t][1]), str(Z[t][2]), str(Z[t][3]))
    print
    '                [%.8s, %.8s, %.8s, %.8s]]' % \
    (str(Z[T - 1][0]), str(Z[T - 1][1]), str(Z[T - 1][2]), str(Z[T - 1][3]))


# --------
#
# The following code checks to see if your particle filter
# localizes the robot to within the desired tolerances
# of the true position. The tolerances are defined at the top.
#

def check_output(final_robot, estimated_position):
    error_x = abs(final_robot.x - estimated_position[0])
    error_y = abs(final_robot.y - estimated_position[1])
    error_orientation = abs(final_robot.orientation - estimated_position[2])
    error_orientation = (error_orientation + pi) % (2.0 * pi) - pi
    correct = error_x < tolerance_xy and error_y < tolerance_xy \
              and error_orientation < tolerance_orientation
    return correct


def dispParticals(p, w, gt):
    xs = np.array([r.x for r in p])
    ys = np.array([r.y for r in p])

    plt.clf()
    lm = np.array(landmarks)
    plt.plot(lm[:, 0], lm[:, 1], 'rX')

    plt.scatter(xs, ys, s=w)

    m_x = np.average(xs, weights=w)
    m_y = np.average(ys, weights=w)
    # m_x, m_y, _ = get_position(p)
    plt.plot(m_x, m_y, 'yX')
    plt.plot(gt[0], gt[1], 'gX')

    plt.legend(['LandMarks','Estimation','GT'],loc=2)
    plt.pause(.1)


def particle_filter(motions, measurements, N=500):  # I know it's tempting, but don't change N!
    # --------
    #
    # Make particles
    #

    p = []
    for i in range(N):
        r = robot()
        r.set_noise(bearing_noise, steering_noise, distance_noise)
        p.append(r)

    # --------
    #
    # Update particles
    #
    for t in range(len(motions)):

        # motion update (prediction)
        p2 = []
        for i in range(N):
            p2.append(p[i].move(motions[t]))
        p = p2

        # measurement update
        w = []
        for i in range(N):
            w.append(p[i].measurement_prob(measurements[t]))

        # resampling
        p3 = []
        index = int(random.random() * N)
        beta = 0.0
        mw = max(w)
        for i in range(N):
            beta += random.random() * 2.0 * mw
            while beta > w[index]:
                beta -= w[index]
                index = (index + 1) % N
            p3.append(p[index])
        p = p3

        w = []
        for i in range(N):
            w.append(p[i].measurement_prob(measurements[t]))
        w = np.array(w)
        w /= w.sum()
        dispParticals(p, w, g_meas[t])
    return get_position(p)


## IMPORTANT: You may uncomment the test cases below to test your code.
## But when you submit this code, your test cases MUST be commented
## out.
##
## You can test whether your particle filter works using the
## function check_output (see test case 2). We will be using a similar
## function. Note: Even for a well-implemented particle filter this
## function occasionally returns False. This is because a particle
## filter is a randomized algorithm. We will be testing your code
## multiple times. Make sure check_output returns True at least 80%
## of the time.

if __name__ == '__main__':
    number_of_iterations = 116
    motions = [[2. * pi / 20, 12.] for row in range(number_of_iterations)]
    motions = [[np.random.random()*2. * pi, np.random.random()*20] for row in range(number_of_iterations)]

    x = generate_ground_truth(motions)
    global g_meas
    g_meas = x[2]
    final_robot = x[0]
    measurements = x[1]
    estimated_position = particle_filter(motions, measurements)
    print_measurements(measurements)
    print('Ground truth:\t\t{}'.format(final_robot))
    print('Particle filter:\t{}'.format(estimated_position))
    print('Code check:\t\t\t{}'.format(check_output(final_robot, estimated_position)))

    plt.show()
