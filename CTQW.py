# Continuous Time Quantum Walk Simulation
# Author: Edmund Dable-Heath
# This is going to be a very rudimentary simulation of a continuous time quantum walk for an arbitrary graph. The idea
# to have as inputs an adjacency matrix, a permeation time and a hopping amplitude. This should fully describe the
# entire simulation. It won't be efficient, I'll work on that afterward.


import numpy as np
from scipy.linalg import expm
from scipy import sparse
import graph_functions as gf


def prop_comp(adjacency_matrix, gamma):
    """
        Computing the propagator unitary matrix for the simulation, considering a fixed prop time and hopping amplitude.
    :param adjacency_matrix: adjacency matrix of the underlying graph
    :param gamma: combination of the hopping amplitude and the propagation time, i.e. the ration between the propagation
                    time and the mass of the walker, float
    :return: np array of unitary matrix propagator
    """
    graph_laplacian = adjacency_matrix - np.diag(np.sum(adjacency_matrix, axis=0))
    return expm(1j * gamma * graph_laplacian)


def prop_comp_sparse (adjacency_matrix, gamma):
    """
         sparse method for above prop comp
    :param adjacency_matrix: scipy sparse version of the adjacency matrix.
    :param gamma: gamma parameter
    :return: np array of unitary matrix propagator
    """
    graph_laplacian = adjacency_matrix - sparse.diags(np.asarray(adjacency_matrix.sum(axis=0))[0])
    return expm(1j * gamma * graph_laplacian)


def ctqw(propagator, initial_position, seed=None):
    """
        continuous time quantum walk on arbitrary graph
    :param propagator: unitary time evolution operator for fixed graph, time, and hopping amplitude.
    :param initial_position: the initial position of the walk, if not given will be uniformly randomly picked.
    :param seed: seed for the randomness
    :return: end state of the walk, ready to be measured and 'collapse' into a single state.
    """
    if not seed:
        seed = np.random.default_rng()
    return seed.choice(propagator.shape[0], p=np.absolute(propagator[initial_position])**2)


if __name__ == "__main__":
    basis = np.array([[32, 0, 0],
                      [1, 1, 0],
                      [1, 0, 1]])
    coord, adj = gf.adj_mat_1d(basis)
    walk = ctqw(prop_comp(adj, 10, 1), 0)
    print(walk)
    print(coord[walk])
