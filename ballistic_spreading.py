# Ballistic Spreading Check
# Author: Edmund Dable-Heath
"""
    The original lattice planar graph CTQW paper was only interested in 2 dimensional analysis. Whilst not explicitly
    stated the key reason this appeared to be the case was for computational ease as simulating the quantum walk for
    much higher dimensions becomes infeasible. Their work implies that a CTQW on a square PLG in 2 dimensions, Z^2,
    exhibits ballistic spreading, i.e. var(Jt) >= (Jt)^2. This feels like it should extend to higher dimension, but
    does it?

    An analytic proof of this would be good as well, but some numerical simulations would be rather easy to achieve so
    those can be run first.
"""

import numpy as np
import CTQW as qw
import graph_functions as gf
import math
import os
import multiprocessing
import networkx as nx


def spatial_var(probability_distribution, coords, coord_choice=0):
    """
        Compute the spatial variance for a given distribution over the integer lattice, based purely on one dimension
        (sufficient due to symmetry):

            sigma_x = <x^2> - <x>^2,       <x> = sum_{j,k=1}^{N_j, N_i} rho_{jk}x_{jk}

    :param probability_distribution: probability distribution over graph, real-(m, )-ndarray
    :param coords: coordinate map from graph to integer lattice, list of lists of ints
    :param coord_choice: coordinate to choice to compute variance over (optional, default=0), int
    :return: spatial variance, float
    """
    if coord_choice > len(coords[0]):
        print('Coordinate choice greater than problem dimension, defaulting to 0')
        coord_choice = 0
    coord_vals = np.asarray([coord[coord_choice] for coord in coords])
    return np.sum(probability_distribution * (coord_vals**2)) - np.sum(probability_distribution * coord_vals)**2


def run(dimension, graph_bounds, max_gamma, gamma_steps):
    """
        Run the ballistic spread experiment for a given dimension, graph bounds and gamma.
    :param dimension: lattice dimension, int
    :param graph_bounds: bounds on the graph, int
    :param max_gamma: maximum gamma to test for, float
    :param gamma_steps: total number of steps to examine gamma for, int
    :return: writes np array of (gamma, spatial var) to file.
    """
    gammas = np.linspace(0, max_gamma, gamma_steps)
    coords, adj = gf.generic_adjacency_matrix(graph_bounds, dimension)
    np.savetxt(
        'spreading_results/'+str(dimension),
        np.asarray([[gamma,
                     spatial_var(
                         np.abs(qw.prop_comp(adj, gamma)[(adj.shape[0]-1)//2])**2,
                         coords
                     )]
                    for gamma in gammas]),
        delimiter=','
    )


def compute_adjacency(dimension, graph_bounds):
    """
        Compute the adjacency matrix of the graph
    """
    g = nx.grid_graph([graph_bounds])


def partial_sum(eigs, start_vecs, gamma):
    """
        Compute the partial sum for the probability density:

            \sum_{s}e^{-i\gamma\lambda_s}V_{ks}

        arguments:  - eigs, eigenvalues of the graph Laplacian - (m, )-ndarray float
                    - start_vecs, starting row of eigenvectors of the graph Laplacian - (m, )-ndarray float
                    - gamma, single system parameter - float
        returns:    - partial sum mentioned above
    """
    s = 0
    for eig_ind in range(eigs.size[0]):
        s += np.exp(-1j*gamma, eigs[eig_ind]) * start_vecs[eig_ind]
    return s


def probability_density(point_index, start_index, vecs, eigs, gamma):
    """
        Compute the full probability density for a particular point given a starting index.

        arguments:  - point_index, point for density - int
                    - start_index, index for the start of the walk - int
                    - vecs, eigenvectors of the graph Laplacian - (m, )-ndarray float
                    - eigs, eigenvalues of the graph Laplacian - (m, )-ndarray float
                    - gamma, single system parameter - float
        returns:    - probability density
    """
    return np.abs(partial_sum(eigs, vecs[start_index], gamma) * np.sum(vecs[point_index]))**2


def spatial_var(coords, vecs, eigs, gamma, start_index=0):
    """
        Compute the spatial variance for a given distribution over the integer lattice, based purely on one dimension
        (sufficient due to symmetry):

            sigma_x = <x^2> - <x>^2,       <x> = sum_{j,k=1}^{N_j, N_i} rho_{jk}x_{jk}

    :param probability_distribution: probability distribution over graph, real-(m, )-ndarray
    :param coords: coordinate map from graph to integer lattice, list of lists of ints
    :param coord_choice: coordinate to choice to compute variance over (optional, default=0), int
    :return: spatial variance, float
    """
    if start_index > len(coords[0]):
        print('Coordinate choice greater than problem dimension, defaulting to 0')
        coord_choice = 0
    coord_vals = np.asarray([coord[coord_choice] for coord in coords])
    prob_dens = np.zeros_like(eigs)
    for i in range(eigs.size[0]):
        prob_dens[i] = probability_density(i, start_index, vecs, eigs, gamma)
    return np.sum(prob_dens * (coord_vals**2)) - np.sum(prob_dens * coord_vals)**2





# Run
if __name__ == "__main__":
    bounds = 9
    max_g = 2.
    g_steps = 20

    for dim in range(2, 8):
        bounds = 5
        run(dim, int(bounds), max_g, g_steps)
        print(f'done with dim {dim}')


