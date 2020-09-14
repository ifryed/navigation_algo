import random

from ParticleLib.utils import *

random.seed(42)


class ParticleFilter:
    particles = []

    def __init__(self, intialX, initialY, std, numOfParticles, landmarks):
        self.number_of_particles = numOfParticles
        self.sigma = std
        self.landarkList = landmarks

        particles = []
        for i in range(self.number_of_particles):
            x = random.gauss(intialX, std)
            y = random.gauss(initialY, std)
            theta = random.uniform(0, 2 * np.pi)
            tmpParticle = np.array([x, y, theta, 1])
            particles.append(tmpParticle)
        self.particles = np.array(particles)

    def moveParticles(self, velocity, yaw_rate, delta_t=0.1):

        theta = self.particles[:, 2]
        new_theta = theta + delta_t * yaw_rate
        new_x = self.particles[:, 0] + (velocity / yaw_rate) * (np.sin(new_theta) - np.sin(theta))
        new_y = self.particles[:, 1] + (velocity / yaw_rate) * (np.cos(theta) - np.cos(new_theta))

        # Add noise
        self.particles[:, 0] = new_x + np.random.normal(0, 0.3, self.number_of_particles)
        self.particles[:, 1] = new_y + np.random.normal(0, 0.3, self.number_of_particles)
        self.particles[:, 2] = new_theta + np.random.normal(0, 0.01, self.number_of_particles)

    def UpdateWeight(self, observations):
        for i, particle in enumerate(self.particles):
            glob_obs = mapObservationsToMapCordinatesList(observations, particle)
            weight = 1

            for glob_ob_single in glob_obs:
                closest_lm = findClosestLandmark(self.landarkList, glob_ob_single)
                obs_prob = findObservationProbability(closest_lm, glob_ob_single, self.sigma, self.sigma)
                weight *= obs_prob
            self.particles[i, 3] = weight

    def getBestParticle(self):
        best_particle_idx = np.argmax(self.particles[:, 3])
        return self.particles[best_particle_idx, :]

    def getBestParticleOut(self):
        best_particle = self.particles.mean(1)
        return best_particle

    def PrintWeights(self):
        for i in range(self.number_of_particles):
            print("Weight:", self.particles[i, 3], self.particles[i, 0], self.particles[i, 1])

    def Resample(self):  # edited
        w = self.particles[:, 3]
        w = w / w.sum()
        max_w_2 = w.max() * 2

        curr_ind = np.random.randint(0, self.number_of_particles)
        new_particles_lst = []
        for i in range(self.number_of_particles):
            offset = np.random.random() * max_w_2

            while offset > w[curr_ind]:
                offset -= w[curr_ind]
                curr_ind = (curr_ind + 1) % self.number_of_particles

            new_particle = np.array([
                self.particles[curr_ind, 0],
                self.particles[curr_ind, 1],
                self.particles[curr_ind, 2],
                self.particles[curr_ind, 3]
            ])
            new_particles_lst.append(new_particle)

        self.particles = np.array(new_particles_lst)
