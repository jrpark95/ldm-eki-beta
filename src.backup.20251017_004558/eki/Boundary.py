import numpy as np


class Boundary(object):
    def __init__(self, boundaryOption):
        self.boundaryOption = boundaryOption

    def outOfBounds(self, particles, bounds):
        lowerBound, upperBound = np.array([bounds]).T
        upper = np.nonzero(particles > upperBound)
        lower = np.nonzero(particles < lowerBound)
        return (lower, upper)

    def nearest(self, particles, bounds):
        lowerBound, upperBound = np.array([bounds]).T
        newParticles = np.where(particles < lowerBound, lowerBound, particles)
        newParticles = np.where(particles > upperBound, upperBound, newParticles)
        return newParticles

    def reflective(self, particles, bounds):
        lowerBound, upperBound = np.array([bounds]).T
        lower, upper = self.outOfBounds(particles, bounds)
        newParticles = particles
        while lower[0].size !=0 or upper[0].size != 0:
            if lower[0].size > 0:
                newParticles[lower] = 2*lowerBound[lower[0]].T - newParticles[lower]
            if upper[0].size > 0:
                newParticles[upper] = 2*upperBound[upper[0]].T-newParticles[upper]
            lower, upper = self.outOfBounds(newParticles, bounds)
        return newParticles

    def random(self, particles, bounds):
        lowerBound, upperBound = np.array([bounds]).T
        lower, upper = self.outOfBounds(particles, bounds)
        newParticles = particles
        newParticles[lower] = (np.array([u-l for u, l in zip(upperBound, lowerBound)])*np.array([np.random.random_sample((particles.shape[0],))]).T + lowerBound)[lower[0]].T[0]
        newParticles[upper] = (np.array([u-l for u, l in zip(upperBound, lowerBound)])*np.array([np.random.random_sample((particles.shape[0],))]).T + lowerBound)[upper[0]].T[0]
        return newParticles

    def periodic(self, particles, bounds):
        lowerBound, upperBound = np.array([bounds]).T
        lower, upper = self.outOfBounds(particles, bounds)
        bound_d = np.tile(np.abs(upperBound - lowerBound), (1, particles.shape[1]))
        lowerBound = np.tile(lowerBound, (1, particles.shape[1]))
        upperBound = np.tile(upperBound, (1, particles.shape[1]))
        newParticles = particles
        if lower[0].size != 0 and lower[1].size !=0:
            newParticles[lower] = upperBound[lower] - np.mod(
                (lowerBound[lower] - newParticles[lower]), bound_d[lower],
            )
        if upper[0].size != 0 and upper[1].size !=0:
            newParticles[upper] = lowerBound[upper] + np.mod(
                (newParticles[upper] - upperBound[upper]), bound_d[upper],
            )
        return newParticles