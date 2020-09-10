from dataclasses import dataclass

import numpy as np


@dataclass
class Particle:
    x: float
    y: float
    theta: float
    weight: float


@dataclass
class LandMark:
    x: float
    y: float
    index: int


def calculateDistance(landmark1, landmark2):
    a = np.sqrt((landmark1.x - landmark2.x) ** 2 + (landmark1.y - landmark2.y) ** 2)
    return a


def findClosestLandmark(map_landmarks, singleObs):  # edited
    # map_landmarks_arr = np.array([[lm.x, lm.y] for lm in map_landmarks])
    # singleObs_arr = np.array([[singleObs.x, singleObs.y]])
    min_index = np.argmin((np.square((map_landmarks - singleObs)[:, :2]).sum(1)))
    closest_landmark = map_landmarks[min_index]

    return closest_landmark


def getError(gt_data, bestParticle):
    error1 = np.abs(gt_data[0] - bestParticle.x)
    error2 = np.abs(gt_data[1] - bestParticle.y)
    error3 = np.abs(gt_data[2] - bestParticle.theta)
    if (error3 > 2 * np.pi):
        error3 = 2 * np.pi - error3
    return (error1, error2, error3)


def findObservationProbability(closest_landmark, map_coordinates, sigmaX, sigmaY):
    mew_x = closest_landmark[0]
    mew_y = closest_landmark[1]

    x = map_coordinates[0]
    y = map_coordinates[1]

    denom = 1 / np.sqrt(sigmaX * sigmaY * 2 * np.pi)
    frac = (1 / (2 * np.pi * sigmaX * sigmaY))
    weight1 = np.square((x - mew_x) / sigmaX) + np.square((y - mew_y) / sigmaY)

    ans = np.exp(-0.5 * weight1) * denom
    return max(ans, np.finfo('float').eps)


def mapObservationToMapCoordinates(observation, particle):
    x = observation[0]
    y = observation[1]

    xt = particle.x
    yt = particle.y
    theta = particle.theta

    MapX = x * np.cos(theta) - y * np.sin(theta) + xt
    MapY = x * np.sin(theta) + y * np.cos(theta) + yt

    return MapX, MapY


def mapObservationsToMapCordinatesList(observations, particle):
    convertedObservations = []
    i = 0
    for obs in observations.iterrows():
        singleObs = obs[1]
        mapX, mapY = mapObservationToMapCoordinates(singleObs, particle)
        tmpLandmark = np.array([mapX, mapY, i])
        i += 1
        convertedObservations.append(tmpLandmark)
    return np.array(convertedObservations)
