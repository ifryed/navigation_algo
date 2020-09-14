import random

from ParticleLib.utils import *

random.seed(42)


class ParticleFilter:
    particles = []

    def __init__(self, intialX, initialY, std, numOfParticles, landmarks):
        self.number_of_particles = numOfParticles
        self.sigma = std
        self.landarkList = landmarks
        for i in range(self.number_of_particles):
            # tmpParticle = Particle(intialX,initialY,0,1)
            x = random.gauss(intialX, std)
            y = random.gauss(initialY, std)
            theta = random.uniform(0, 2 * np.pi)
            tmpParticle = Particle(x, y, theta, 1)
            self.particles.append(tmpParticle)

    def moveParticles(self, velocity, yaw_rate, delta_t=0.1):

        for i in range(self.number_of_particles):
            if (yaw_rate != 0):
                theta = self.particles[i].theta
                newTheta = theta + delta_t * yaw_rate;
                newX = self.particles[i].x + (velocity / yaw_rate) * (np.sin(newTheta) - np.sin(theta));
                newY = self.particles[i].y + (velocity / yaw_rate) * (np.cos(theta) - np.cos(newTheta));

                # todo Add noise!!
                self.particles[i].x = newX + random.gauss(0, 0.3)
                self.particles[i].y = newY + random.gauss(0, 0.3)
                self.particles[i].theta = newTheta + random.gauss(0, 0.01)
            else:
                print("ZERO!!!")

    def UpdateWeight(self, observations):
        for i, particle in enumerate(self.particles):
            glob_obs = mapObservationsToMapCordinatesList(observations, particle)
            weight = 1

            for glob_ob_single in glob_obs:
                closest_lm = findClosestLandmark(self.landarkList, glob_ob_single)
                obs_prob = findObservationProbability(closest_lm, glob_ob_single, self.sigma, self.sigma)
                weight *= obs_prob
            self.particles[i].weight = weight

    def getBestParticle(self):
        best_particle = max(self.particles, key=lambda particle: particle.weight)
        return best_particle

    def getBestParticleOut(self):
        x = 0
        y = 0
        theta = 0
        for i in range(self.number_of_particles):
            x += self.particles[i].x
            y += self.particles[i].y
            theta += self.particles[i].theta
        x = x / self.number_of_particles
        y = y / self.number_of_particles
        theta = theta / self.number_of_particles
        best_particle = Particle(x, y, theta, weight=1)
        return best_particle

    def PrintWeights(self):
        for i in range(self.number_of_particles):
            print("Weight:", self.particles[i].weight, self.particles[i].x, self.particles[i].y)

    def Resample(self):  # edited
        w = np.array([x.weight for x in self.particles])
        w = w / w.sum()
        max_w_2 = w.max() * 2

        curr_ind = np.random.randint(0, self.number_of_particles)
        new_particles_lst = []
        for i in range(self.number_of_particles):
            offset = np.random.random() * max_w_2

            while offset > w[curr_ind]:
                offset -= w[curr_ind]
                curr_ind = (curr_ind + 1) % self.number_of_particles

            new_particle = Particle(
                self.particles[curr_ind].x,
                self.particles[curr_ind].y,
                self.particles[curr_ind].theta,
                self.particles[curr_ind].weight
            )
            new_particles_lst.append(new_particle)

        self.particles = list(new_particles_lst)
