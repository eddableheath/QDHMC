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
    dist = np.asarray(
        np.abs(
            expm(1j * gamma * l.tocsc())[l.shape[0] // 2]
        ).todense()
    )[0]**2
    coords = [[node[index] - 9 for index in range(dim)]
              for node in nx.nodes(g)]
    np.savetxt(
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


def alt_compute_dist(gamma, dim, bounds):
    g = nx.grid_graph([bounds for _ in range(dim)])
    a = nx.to_numpy_array(g)
    d, v = np.linalg.eig(
        a - np.diag(
            np.sum(a, axis=0)
        )
    )
    dist = np.abs(
        np.asarray(
            [
                np.sum(
                    [
                        v[i, k] * np.exp(1j * gamma * d[k]) * v[v.shape[0] // 2, k] for k in range(v.shape[0])
                    ]
                ) for i in range(v.shape[0])
            ]
        )
    ) ** 2
    dist = dist / sum(dist)
    coords = [[node[index] - bounds for index in range(dim)]
              for node in nx.nodes(g)]
    np.savetxt(
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
    alt_compute_dist(
        gamma,
        pars['dimension'],
        pars['bounds']
    )


if __name__ == "__main__":
    spec_pars = {
        'gamma': 1,
        'dimension': 5,
        'bounds': 9
    }
    main(spec_pars)
