"""
Boundary Handling Utilities for Ensemble States

This module provides various boundary constraint handling methods for ensemble
optimization. Useful for enforcing box constraints on state variables (e.g.,
non-negative emission rates).

Supported boundary handling methods:
    - nearest: Clamp to nearest boundary
    - reflective: Mirror reflection at boundaries
    - random: Random resampling within bounds
    - periodic: Periodic (toroidal) boundaries

Author:
    Siho Jang, 2025

Notes:
    These methods are optional and not currently used in the main EKI workflow.
    The REnKF method provides an alternative constraint handling approach via
    penalty functions.
"""

import numpy as np


class Boundary(object):
    """
    Boundary constraint handler for ensemble states.

    Provides multiple strategies for handling state variables that violate
    box constraints during ensemble updates.

    Parameters
    ----------
    boundaryOption : str
        Name of boundary handling method ('nearest', 'reflective', 'random', 'periodic')

    Attributes
    ----------
    boundaryOption : str
        Selected boundary handling method
    """
    def __init__(self, boundaryOption):
        self.boundaryOption = boundaryOption

    def outOfBounds(self, particles, bounds):
        """
        Identify particles that violate bounds.

        Parameters
        ----------
        particles : ndarray of shape (ndim, nparticles)
            Particle/ensemble member positions
        bounds : ndarray of shape (ndim, 2)
            Lower and upper bounds for each dimension

        Returns
        -------
        lower : tuple of ndarrays
            Indices of particles below lower bound
        upper : tuple of ndarrays
            Indices of particles above upper bound
        """
        lowerBound, upperBound = np.array([bounds]).T
        upper = np.nonzero(particles > upperBound)
        lower = np.nonzero(particles < lowerBound)
        return (lower, upper)

    def nearest(self, particles, bounds):
        """
        Clamp out-of-bounds particles to nearest boundary.

        Parameters
        ----------
        particles : ndarray of shape (ndim, nparticles)
            Particle positions
        bounds : ndarray of shape (ndim, 2)
            Lower and upper bounds

        Returns
        -------
        newParticles : ndarray of shape (ndim, nparticles)
            Particles with violations clamped to bounds
        """
        lowerBound, upperBound = np.array([bounds]).T
        newParticles = np.where(particles < lowerBound, lowerBound, particles)
        newParticles = np.where(particles > upperBound, upperBound, newParticles)
        return newParticles

    def reflective(self, particles, bounds):
        """
        Mirror-reflect particles at boundaries.

        Particles that cross a boundary are reflected back into the feasible
        region. Multiple reflections are handled iteratively.

        Parameters
        ----------
        particles : ndarray of shape (ndim, nparticles)
            Particle positions
        bounds : ndarray of shape (ndim, 2)
            Lower and upper bounds

        Returns
        -------
        newParticles : ndarray of shape (ndim, nparticles)
            Particles after reflective boundary handling
        """
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
        """
        Resample out-of-bounds particles uniformly within bounds.

        Parameters
        ----------
        particles : ndarray of shape (ndim, nparticles)
            Particle positions
        bounds : ndarray of shape (ndim, 2)
            Lower and upper bounds

        Returns
        -------
        newParticles : ndarray of shape (ndim, nparticles)
            Particles with violations replaced by random samples
        """
        lowerBound, upperBound = np.array([bounds]).T
        lower, upper = self.outOfBounds(particles, bounds)
        newParticles = particles
        newParticles[lower] = (np.array([u-l for u, l in zip(upperBound, lowerBound)])*np.array([np.random.random_sample((particles.shape[0],))]).T + lowerBound)[lower[0]].T[0]
        newParticles[upper] = (np.array([u-l for u, l in zip(upperBound, lowerBound)])*np.array([np.random.random_sample((particles.shape[0],))]).T + lowerBound)[upper[0]].T[0]
        return newParticles

    def periodic(self, particles, bounds):
        """
        Apply periodic (toroidal) boundary conditions.

        Particles wrap around to the opposite boundary when they cross.

        Parameters
        ----------
        particles : ndarray of shape (ndim, nparticles)
            Particle positions
        bounds : ndarray of shape (ndim, 2)
            Lower and upper bounds

        Returns
        -------
        newParticles : ndarray of shape (ndim, nparticles)
            Particles after periodic wrapping
        """
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