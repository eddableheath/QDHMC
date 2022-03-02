# Functions for Graph Theory Parts
# Author: Edmund Dable-Heath
# This will contain all the function necessary to compute the adjacency matrix for the truncated square lattice for a
# given basis. For a given basis we have many given bounds on the size of the lattice necessary for the search, so the
# basis will functionally define the whole system.


import numpy as np
import networkx as nx
import math


def adjacency_matrix(lattice_basis, walk_dimension=None):
    """
        Given a lattice basis B return the adjacency matrix for the truncated section of the integers lattice which
        contains the shortest vector.
    :param lattice_basis: lattice basis, np array
    :param walk_dimension: stated dimension for walk on sub dims of full lattice but requiring integer bounds from lattice
    :return: adjacency matrix.
    """
    lattice_dimension = lattice_basis.shape[0]
    if not walk_dimension:
        walk_dimension = lattice_basis.shape[0]
    integer_bounds = math.ceil(lattice_dimension*math.log2(lattice_dimension) + math.log2(np.linalg.det(lattice_basis)))
    grid_graph = nx.grid_graph(dim=[2*integer_bounds
                                    for i in range(walk_dimension)])
    recentred_nodes_coordinates = [[node[index] - integer_bounds
                                    for index in range(walk_dimension)]
                                   for node in nx.nodes(grid_graph)]
    return recentred_nodes_coordinates, nx.to_numpy_array(grid_graph)


def generic_adjacency_matrix(graph_bounds, graph_dimension):
    """
        For a generic square planar lattice graph in a specified dimension over specified range compute the adjacency
        matrix and output the mapping back to Z^n centred around the origin.
    :param graph_bounds: bounds on the graph, int
    :param graph_dimension: dimension of the graph, int
    :return: - mapping back to Z^n centred around origin, ndarray
             - adjacency matrix, ndarray
    """
    grid_graph = nx.grid_graph(dim=[2*graph_bounds + 1
                                    for i in range(graph_dimension)])
    return ([[node[index] - graph_bounds
             for index in range(graph_dimension)]
            for node in nx.nodes(grid_graph)],
            nx.to_numpy_array(grid_graph))


def generic_adjacency_sparse (graph_bounds, graph_dimension):
    """
        Outputs a scipy.sparse matrix for the adjacency matrix instead.
    :param graph_bounds: range of points on the graph
    :param graph_dimension: dimension of the graph
    :return: sparse adjacency matrix and coords
    """
    grid_graph = nx.grid_graph(dim=[2*graph_bounds +1
                                    for i in range(graph_dimension)])
    return ([[node[index] - graph_bounds
              for index in range(graph_dimension)]
             for node in nx.nodes(grid_graph)],
            nx.to_scipy_sparse_matrix(grid_graph))


def adj_mat_1d(lattice_basis):
    """
         Given a lattice basis return a one dimensional quantum walk over one of the dimensions.
    :param lattice_basis:  lattice basis, np array, purely for the bounds of the array.
    :return: adjacency matrix and coordinates
    """
    lattice_dimension = lattice_basis.shape[0]
    integer_bounds = math.ceil(lattice_dimension*math.log2(lattice_dimension) + math.log2(np.linalg.det(lattice_basis)))
    grid_graph = nx.grid_graph(dim=[2*integer_bounds])
    centered_nodes = [i - integer_bounds for i in [node for node in nx.nodes(grid_graph)]]
    return centered_nodes, nx.to_numpy_array(grid_graph)


if __name__ == "__main__":
    basis = np.array([[32, 0, 0],
                      [1, 1, 0],
                      [1, 0, 1]])
    coords, amat = adj_mat_1d(basis)
    print(coords)