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

# Run
if __name__ == "__main__":
    dim_2_range = 10
    max_g = 2.
    g_steps = 20

    for dim in range(3, 8):
        bounds = (dim_2_range**2)**(1/float(dim)) // 2 - 1
        run(dim, int(bounds), max_g, g_steps)
        print(f'done with dim {dim}')


