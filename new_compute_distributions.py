# Computing Bigger distributions
# Author: Edmund Dable-Heath
"""
    Previous method for computing distributions scales horrifically in memory so can't be done only in memory. Needs to
    either compute in place in storage or use some other kind of method.
"""

import numpy as np
import networkx as nx
from scipy.linalg import expm
import multiprocessing as mp


def compute_dist(gamma, dim):
    g = nx.grid_graph([9 for _ in range(dim)])
    l = - nx.linalg.laplacian_matrix(g)
    dist = expm(1j * gamma * l)[l.shape[0] // 2]
    coords = [[node[index] - 9 for index in range(dim)]
              for node in nx.nodes(g)]
    np.save_txt(
        f'distributions/dist_{dim}.csv',
        dist,
        delimiter=','
    )
    np.savetxt(
        f'distribution/coords_{dim}.csv',
        coords,
        delimiter=',',
        fmt='%5.0f'
    )


def main(pars):
    gamma = pars['gamma'] * 0.5
    cores = mp.cpu_count() // 2
    pool = mp.Pool(cores)
    [
        pool.apply(compute_dist,
                   args=(gamma, dimension))
        for dimension in range(2, 6)
    ]
    pool.join()
    pool.close()


if __name__ == "__main__":
    spec_pars = {
        'gamma': 1
    }
    main(spec_pars)
