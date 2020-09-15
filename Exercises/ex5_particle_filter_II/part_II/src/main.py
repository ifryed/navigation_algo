import pandas as pd
import numpy as np
import time
import glob

from src.ParticleLib import ParticleFilter
from src.ParticleLib import utils as pUtils
import matplotlib.pyplot as plt

np.random.seed(14)


def loadData(data_path="../data/"):
    gt_data = pd.read_csv(data_path + 'gt_data.txt', names=['X', 'Y', 'Orientation'], sep=' ')
    map_data = pd.read_csv(data_path + 'map_data.txt', names=['X', 'Y', '# landmark'])
    control_data = pd.read_csv(data_path + 'control_data.txt', names=['velocity', 'Yaw rate'], sep=' ')

    # observation = pd.read_csv('data/observation/observations_000001.txt', names = ['X cord','Y cord'], sep=' ')

    result = [(x, y, landmark) for x, y, landmark in zip(map_data['X'], map_data['Y'], map_data['# landmark'])]
    landarkList = []
    for res in result:
        # l = pUtils.LandMark(res[0], res[1], res[2])
        landarkList.append((res[0], res[1], res[2]))
    landarkList = np.array(landarkList)

    obs_path = glob.glob(data_path + "/SensorDataFiles/observation*.txt")
    obs_path.sort()

    print('Loading the Observations..')
    observation = []
    for file_path in obs_path:
        observationTmp = pd.read_csv(file_path, names=['X cord', 'Y cord'], sep=' ')
        observation.append(observationTmp)
    print('Loading Done!')
    return observation, control_data, gt_data, landarkList


sigmaY = 0.3
magicNumberOfParticles = 200


def displaytrajectory(particles, bestP, gt, plot_all=True, disp=True):
    prts = np.array(particles)
    prts[:, 2] /= prts[:, 2].sum()

    if plot_all:
        plt.scatter(prts[:, 0], prts[:, 1], c=prts[:, 2], cmap='gray', s=10)
    plt.plot(gt[0], gt[1], 'g*')
    plt.plot(bestP[0], bestP[1], 'c*')

    if disp:
        plt.pause(0.1)


def main():
    observation, control_data, gt_data, landmarks = loadData()
    particleFilter = ParticleFilter.ParticleFilter(0, 0, .6, numOfParticles=magicNumberOfParticles,
                                                   landmarks=landmarks)

    for i, obs in enumerate(observation):
        # prediction
        if i != 0:
            velocity = control_data.iloc[i - 1][0]
            yaw_rate = control_data.iloc[i - 1][1]
            particleFilter.moveParticles(velocity, yaw_rate)

        particleFilter.UpdateWeight(np.array(obs))
        particleFilter.Resample()
        bestP = particleFilter.getBestParticle()
        error = pUtils.getError(gt_data.iloc[i], bestP)
        print(i, error)

        displaytrajectory(particleFilter.particles, bestP, gt_data.iloc[i], i % 40 == 0, False)

    plt.savefig('../results .png')
    # plt.show()


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print("Time: {:.3f} secs".format(end - start))
