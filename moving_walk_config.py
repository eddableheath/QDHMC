# Config file for CTQW moving run
# Author: Edmund Dable-Heath
"""
    The config file for the basic continuous time quantum walk experiment.
"""

import numpy as np
import math


def compute_graph_bounds(lattice_basis, total_points):
    """
        Compute the range that the graph walk should be to be reasonably computable. Given a total number of points it
        will give bounds that have either approximately those number of points spread over the correct dimensions or
        the full integer bounds.
    :param lattice_basis: lattice basis for computation of the dimension and integer bounds, int (m,m)-ndarray
    :param total_points: total points to be included, int
    :return: maximum bounds for the graph in each dimension, int
    """
    dimension = lattice_basis.shape[0]
    integer_bounds = math.ceil(dimension * math.log2(dimension) + math.log2(np.linalg.norm(lattice_basis)))
    bound = math.floor(total_points ** (1/float(dimension)) / 2)
    if bound > integer_bounds:
        return integer_bounds
    else:
        return bound


# Multiprocessing parameters
cores = 4

# Lattice parameters ---------------------------------
dimension = 2
lattice_num = 0
lattice_basis = np.genfromtxt(
    'run_data/lattice.csv',
    delimiter=',',
    dtype=None
)

# Walk parameters
graph_bounds = compute_graph_bounds(lattice_basis, 31**2)
dist = np.genfromtxt(
    'run_data/dist_2.csv',
    delimiter=',',
    dtype=None
)
coords = np.genfromtxt(
    'run_data/coords_2.csv',
    delimiter=',',
    dtype=None
)
coords = coords.astype(int)

# Model parameters -----------------------------------
gamma_mark = 1

number_of_runs = 1
