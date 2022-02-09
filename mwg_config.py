# Config file for MWG walk
# Author: Edmund Dable-Heath
"""
    Config file for the metropolis within gibbs approach to the quantum walk SVP experiment.
"""

import numpy as np
import math


def compute_graph_bounds(basis, t_points, sub_dim):
    """
        Compute the range that the graph walk should be constrained to for either computational efficacy reasons or to
        ensure the existence of the shortest vector within the scope.
    :param basis: lattice basis for computation of the dimension and the integer bounds, int-(m,m)-ndarray
    :param t_points: the total points to be considered for computation efficacy, int
    :param sub_dim: the dimension of the current walk being considered, either 2 or 3 dimension, int
    """
    dim = basis.shape[0]
    integer_bounds = math.ceil(dim * math.log2(dim) + math.log2(np.linalg.det(basis)))
    bound = math.floor(t_points ** (1/float(sub_dim)) / 2)
    if bound > integer_bounds:
        return integer_bounds
    else:
        return  bound


# Multiprocessing parameters
cores = 4

# Lattice parameters ----------------------
dimension = 5
lattice_type = 'hnf'
lattice_num = 0
lattice_basis = np.genfromtxt(
    'run_data/lattice.csv',
    delimiter=',',
    dtype=None
)

# Walk parameters -------------------------
total_points = 31**2
graph_bounds_2 = compute_graph_bounds(lattice_basis, total_points, 2)
graph_bounds_3 = compute_graph_bounds(lattice_basis, total_points, 3)
dist_2 = np.genfromtxt(
    'run_data/dist_2.csv',
    delimiter=',',
    dtype=None
)
dist_3 = np.genfromtxt(
    'run_data/dist_3.csv',
    delimiter=',',
    dtype=None
)
coords_2 = np.genfromtxt(
    'run_data/coords_2.csv',
    delimiter=',',
    dtype=None
)
coords_3 = np.genfromtxt(
    'run_data/coords_3.csv',
    delimiter=',',
    dtype=None
)

# Model Parameter -------------------------
gamma_mark = 1
number_of_runs = 1
