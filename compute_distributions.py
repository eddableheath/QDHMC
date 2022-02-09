# Computing Quantum Walk Distributions
# Author: Edmund Dable-Heath
"""
    In the new scheme for each dimension I'm using the same distribution each time but also recomputing it each time.
    By precomputing the distribution I can save a lot of time, and space.
"""

import numpy as np
import multiprocessing as mp
import graph_functions as gf
import CTQW as qf
import math


def gen_generic_prob_density(dimension, graph_bounds, gamma):
    """
        For a generic walk on a square planar lattice graph in a specified dimension or specified range and propagation
        time output the probability density generated from the walker starting at the central node.
    :param dimension: dimension of the graph to be searched over.
    :param graph_bounds: bounds of the graph to be searched over.
    :param gamma: single parameter for this model, the ration of the propagation time to the mass of the walker.
    :return: (m,) real ndarray representing the probability.
    """
    coords, adj_mat = gf.generic_adjacency_matrix(graph_bounds, dimension)
    return coords, np.absolute(qf.prop_comp(adj_mat, gamma)[(adj_mat.shape[0]-1)//2])**2


def compute_graph_bounds(lattice_basis, total_points, sub_dim):
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
    bound = math.floor(total_points ** (1/float(sub_dim)) / 2)
    if bound > integer_bounds:
        return integer_bounds
    else:
        return bound


def write_dist(dimension, gamma_marker):
    """
        For a given dimension compute the distribution of a quantum walk of on a square PLG of given bounds, from zero.
            - The arbitrary graph is to make the code run more easily. All it contains really is the dimension and
                covolume information. The covolume is set at 32 for all lattices currently. If that changes this will
                need to change.
            - 31**2 for the choice of max number of points is justified computationally (or so I say....).
    :param dimension: dimension of the square PLG
    :param gamma_marker: gamma parameter, controlling the propagation time of the walk, this is the marker to more
                            easily store the data. Will be multiplied by 0.5.
    :return: real-(m, )-ndarray containing the distribution. m here is the number of vertices in the graph.
    """
    arbitrary_basis = np.eye(dimension)
    arbitrary_basis[0][0] = 32
    coords, dist = gen_generic_prob_density(
        dimension,
        compute_graph_bounds(
            arbitrary_basis,
            31**2, 2
        ),
        gamma_marker * 0.5
    )
    np.savetxt(
        'distributions/dist_'+str(dimension)+'_2.csv',
        dist,
        delimiter=','
    )
    np.savetxt(
        'distributions/coords_'+str(dimension)+'_2.csv',
        coords,
        delimiter=','
    )
    coords, dist = gen_generic_prob_density(
        dimension,
        compute_graph_bounds(
            arbitrary_basis,
            31 ** 2, 3
        ),
        gamma_marker * 0.5
    )
    np.savetxt(
        'distributions/dist_' + str(dimension) + '_3.csv',
        dist,
        delimiter=','
    )
    np.savetxt(
        'distributions/coords_' + str(dimension) + '_3.csv',
        coords,
        delimiter=','
    )


# Run
if __name__ == "__main__":
    pars = {
        'cores': 32,
        'min_dim': 2,
        'max_dim': 7,
        'gamma_marker': 1
    }

    pool = mp.Pool(pars['cores'])

    [pool.apply(
        write_dist,
        args=(
            dim,
            pars['gamma_marker']
        )
    ) for dim in range(pars['min_dim'], pars['max_dim']+1)]
