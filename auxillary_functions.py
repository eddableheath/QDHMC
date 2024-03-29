# Functions for main file
# Author: Edmund Dable-Heath
# Here are the various functions necessary for the main implementation of the lattice-discontinuous-hamiltonian-MC over
# the integers.

import numpy as np
import math
import klein_sampler as ks
import random as rn


# Functions for the main algorithm -------------------------------------------------------------------------------------
def potential_energy(basis, point, zero_point_energy):
    """
    computing the potential energy for an intger point, generalised to the neighbourhood of an integer
    (the voronoi cell around the point)
    :param basis: lattice basis, a numpy array of integers
    :param point: point from algorithm, numpy array, usually integers
    :param zero_point_energy: relatively arbitrary zero point energy to remove zero from equation
    :return: the potential energy associated with the particular point
    """
    lattice_point = np.dot(basis.T, np.around(point))
    point_norm = np.linalg.norm(lattice_point)
    return abs(point_norm)
    # if abs(point_norm) > 0:
    #     return abs(math.log2(point_norm))
    # else:
    #     return abs(math.log2(zero_point_energy))


def non_zero_init_point(basis):
    """
    using klein sampler to pick a nonzero starting vector
    :param basis: lattice basis, numpy array
    :return: integer coefficients for the picked lattice vector
    """
    while True:
        klein_sample = ks.klein_sampler(basis, 64)
        if np.linalg.norm(klein_sample) > 0:
            return np.linalg.solve(basis.T, klein_sample)
        else:
            continue


# Functions for post  processing ---------------------------------------------------------------------------------------
def find_min_vector(vectors):
    """
    Given an array of 2D arrays, find which of the sub arrays has the minimum euclidean norm
    :param vectors: array of 2D arrays
    :return: minimum array
    """
    unique_vectors = np.unique(vectors, axis=1)
    return unique_vectors[np.argmin(np.linalg.norm(unique_vectors, axis=1))]


# Misc extra functions -------------------------------------------------------------------------------------------------
def range_calc(basis):
    """
    computing the integer range to ensure the shortest vector is within the scope - for visualisation
    :param basis: lattice basis, numpy array
    :return: return bound on integer range such that the shortest vector is assured to be within the enumerated space
    """
    dimension = basis.shape[0]
    return dimension * math.log2(dimension) + math.log2(abs(np.linalg.det(basis)))


# Metropolis Filter-----------------------------------------------------------------------------------------------------
def metropolis_filter(current_state, proposal_state, lattice_basis, target_function, propagator):
    """
        A metropolis filter return a simple bool for accept or reject based on give function and proposal state.
    :param current_state: current state of the markov chain, tuple of integer point and graph numbering
    :param proposal_state: proposal state generated by the quantum walk algorithm, tuple as above
    :param lattice_basis: the basis of the number theoretic lattice
    :param target_function: what function is being used to define the metropolis filter.
    :param propagator: the proposal distribution from the quantum walk.
    :return: Bool for accept or reject proposal step
    """
    backward_prop = np.absolute(propagator[proposal_state[1]])**2
    forward_prop = np.absolute(propagator[current_state[1]])**2
    alpha = min(1,
                (target_function(lattice_basis, current_state[0], proposal_state[0]))
                * (backward_prop[current_state[1]] / forward_prop[proposal_state[1]]))
    if np.random.uniform(0, 1) <= alpha:
        return True
    else:
        return False


def metropolis_filter_simple(current_state, proposal_state, lattice_basis, sigma):
    """
        A metropolis filter return a simple bool for accept or reject based on give function and proposal state, simpler
        version ignoring propsoal function.
    :param current_state: current state of the markov chain, tuple of integer point and graph numbering
    :param proposal_state: proposal state generated by the quantum walk algorithm, tuple as above
    :param lattice_basis: the basis of the number theoretic lattice
    :param sigma: the standard deviation for the lattice gaussian, float
    :return: Bool for accept or reject proposal step
    """
    alpha = min(1, (lattice_gaussian(lattice_basis, current_state, proposal_state, sigma)))
    if np.random.uniform(0, 1) <= alpha:
        return True
    else:
        return False


def metropolis_filter_log_cost(current_state, proposal_state, lattice_basis):
    """
        Alternative Metropolis filter with log based cost function (not a well defined distribution so this is more like
        a SGD algorithm):
                |log(||Bx||)|
        This makes the length of the shortest vector the global minimum in integral lattices.
    :param current_state: current integer vector state, int (m, )-ndarray
    :param proposal_state: proposed integer vector state, int (m, )-ndarray
    :param lattice_basis: lattice basis, int (m, m)-ndarray
    :return: acceptance or rejection of proposal state, Bool
    """
    alpha = min(1,
                np.where(np.linalg.norm(lattice_basis.T @ proposal_state) != 1,
                         np.where(np.linalg.norm(lattice_basis.T @ proposal_state) != 0,
                                  np.log(np.linalg.norm(lattice_basis.T @ current_state)) /
                                  np.log(np.linalg.norm(lattice_basis.T @ proposal_state)),
                                  0),
                         np.inf)
                )
    if np.random.uniform(0, 1) <= alpha:
        return True
    else:
        return False


# Lattice Gaussian------------------------------------------------------------------------------------------------------
def lattice_gaussian(lattice_basis, old_vector, new_vector, sigma):
    """
        Implementation of a lattice gaussian function
    :param lattice_basis: basis for the lattice
    :param old_vector: previous vector in markov chain, for comparison
    :param new_vector: current vector for finding the gaussian value
    :param sigma: standard deviation of lattice gaussian, float
    :return: gaussian density value for given vector
    """
    factor = -(np.linalg.norm(np.dot(lattice_basis.T, new_vector))**2
               - np.linalg.norm(np.dot(lattice_basis.T, old_vector))**2)/(2 * sigma**2)
    try:
        ans = math.exp(factor)
    except OverflowError:
        ans = float('inf')
    return ans


# Functions for met within Gibbs adaptation ----------------------------------------------------------------------------
def split_dim_3(dimension, random=False):
    """
        split the dimension into a dictionary with the correct split of 2 and 3 subdims.
    :param dimension: dimension of the lattice problem.
    :return: dictionary with relevant indices for subdims.
    """
    if not random:
        if dimension % 2 == 0:
            return [(i*2, i*2+1) for i in range(int(dimension/2))]
        else:
            inds = [(i*2, i*2+1) for i in range(int(dimension/2) - 1)]
            inds.append((dimension-3, dimension-2, dimension-1))
            return inds
    else:
        indices = [i for i in range(dimension)]
        rn.shuffle(indices)
        if dimension % 2 == 0:
            return [(indices[i*2], indices[i*2+1]) for i in range(int(dimension/2))]
        else:
            inds = [(indices[i*2], indices[i*2+1]) for i in range(int(dimension/2)-1)]
            inds.append((indices[-3], indices[-2], indices[-1]))
            return inds


def split_dim(dimension, random=False):
    """
        split the dimension into a dictionary with the correct split of 2 and 1 subdims.
    :param dimension: dimension of the lattice problem.
    :param random: whether to randomly split up the dimensions
    :return: list of tuples with relevant indices for subdims.
    """
    if not random:
        if dimension % 2 == 0:
            return [(i*2, i*2+1) for i in range(int(dimension/2))]
        else:
            inds = [(i*2, i*2+1) for i in range(int(dimension/2))]
            inds.append((dimension-1))
            return inds
    else:
        indices = [i for i in range(dimension)]
        rn.shuffle(indices)
        if dimension % 2 == 0:
            return [(indices[i*2], indices[i*2+1]) for i in range(int(dimension/2))]
        else:
            inds = [(indices[i*2], indices[i*2+1]) for i in range(int(dimension/2))]
            inds.append(indices[-1])
            return inds


# Testing
if __name__ == "__main__":
    x=1
    # basis = np.array([[32, 0],
    #                   [10, 1]])
    # sv = np.array([2, -3])
    # prop = np.array([4, 3])
    # cur = np.array([1, -2])
    # alpha = min(1, np.where(np.linalg.norm(sv) != 1,
    #                         np.where(np.linalg.norm(sv) != 0,
    #                                  np.log(np.linalg.norm(basis @ cur)) /
    #                                  np.log(np.linalg.norm(sv)),
    #                                  0),
    #                         np.inf))
    # print(np.linalg.norm(sv))
    # print(np.linalg.norm(basis@cur))
    # print(alpha)
