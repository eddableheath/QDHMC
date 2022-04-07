# Precompute diagonalised graph laplacians
# Author: Edmund Dable-Heath
"""
    For the purposes of examining the ballistic spread of the quantum walk on a planar lattice graph in n dimensions an
    alternative method of computation for the propagator is necessary that requires a precomputation of the graph
    laplacian and subsequent diagonalisation.
"""

import numpy as np
import networkx as nx
import multiprocessing as mp


def diag_laplacian(dimension, bounds):
    """
        Given a dimension and bounds returns the diagonalisation of the laplacian for a planar lattice graph, as well as
        the relevant coordinates.
    dimension: (d) dimension of the graph, int
    bounds: (b) bounds on the size of the graph in every dimension, odd to be centered around zero, int

    returns:   D, V, coords
                D - eigenvalues of graph laplacian, float (d^b, )-ndarray
                V - eigenvectors of graph laplacian, float (d^b, d^b)-ndarray
                coords - list of coordinates for each node shifted to be centered around zero.
    """
    if bounds % 2 == 0:
        bounds += 1
    g = nx.grid_graph([bounds for _ in range(dimension)])
    a = nx.to_numpy_array(g)
    d, v = np.linalg.eigh(
        a - np.diag(np.sum(a, axis=0))
    )



def main():
    return 1


if __name__ == "__main__":
    pars = {
        'write_path': '/some path I geuss'
    }
    main()
