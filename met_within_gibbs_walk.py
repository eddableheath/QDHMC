# Metropolis-Within-Gibbs Quantum Walk for SVP
# Author: Edmund Dable-Heath
"""
    Refactored approach to the metropolis-within-gibbs approach to the quantum walk SVP algorithm. This takes a random
    two dimensions and applies the quantum walk to them, accepting the new point based on a Metropolis filter comparing
    the full rank original vector with the new vector with updated sub coordinates.

    For odd dimensions this will perform a three-dimensional on the remaining three dimensions.
            - In the future this could be compared to a one dimensional walk for the overflow dimensions.

    Each run it will randomly reorganise the dimensions into pairs + one triple.
"""

import mwg_config as config
import numpy as np
from copy import copy
from random import shuffle
import auxillary_functions as af
import os
import multiprocessing as mp
from re import findall
from itertools import filterfalse


class WalkExperiment:

    """
        Class for the walk experiment.
    """

    def __init__(
            self, lattice_basis, prop_density_2, prop_density_3, graph_coords_2, graph_coords_3, graph_bounds_2,
            graph_bounds_3, markov_iters, latt_gaus_sigma=None
    ):
        self.basis = lattice_basis
        self.dimension = self.basis.shape[0]
        self.int_lattice_bounds = self.dimension * np.log2(self.dimension) + np.log2(np.linalg.det(self.basis))

        self.prob_density_2 = prop_density_2
        self.prob_density_3 = prop_density_3
        self.graph_coords_2 = graph_coords_2
        self.graph_coords_3 = graph_coords_3
        self.graph_bounds_2 = graph_bounds_2
        self.graph_bounds_3 = graph_bounds_3

        if latt_gaus_sigma is not None:
            self.latt_gauss_sigma = latt_gaus_sigma
        else:
            self.latt_gauss_sigma = np.sqrt(self.dimension) * \
                                    np.abs(np.linalg.det(self.basis)**(1/float(self.dimension)))

        self.current_integer_vector = np.random.randint(-self.graph_bounds_2, self.graph_bounds_2, self.dimension)
        self.markov_chain = [copy(self.current_integer_vector)]
        self.lattice_markov_chain = [self.basis.T @ copy(self.current_integer_vector)]
        self.markov_iters = markov_iters

        self.current_dimension_splits = []
        self.update_dimension_splits()

    def update_dimension_splits(self):
        """
            Update the splits of the sub-dimensions, reshuffle into groups of two and one three.
        """
        dims = [i for i in range(self.dimension)]
        shuffle(dims)
        n = self.dimension // 2
        sub_dims = [dims[i::n] for i in range(n)]
        shuffle(sub_dims)
        self.current_dimension_splits = sub_dims

    def gibbs_update(self, prop_state, sub_dim):
        """
            Gibbs update of the vector for a given set of sub-dims.
        :param prop_state: the proposal state to update, int-(self.dim, self.dim)-ndarray
        :param sub_dim: list of the current sub-dimensions to be changed, list of ints
        """
        sub_d = len(sub_dim)
        if sub_d == 2:
            prop_alt = self.graph_coords_2[np.random.choice([i for i in range(self.prob_density_2.shape[0])],
                                                            p=self.prob_density_2.tolist())]
        elif sub_d == 3:
            prop_alt = self.graph_coords_3[np.random.choice([i for i in range(self.prob_density_3.shape[0])],
                                                            p=self.prob_density_3.tolist())]
        else:
            print('How has this happened, it is broken mate.')
        for i in range(sub_d):
            prop_state[sub_dim[i]] = prop_state[sub_dim[i]] + prop_alt[i]
        return prop_state

    def update_state(self):
        """
            Update full vector by applying gibbs updates
        """
        for sub_dim in self.current_dimension_splits:
            proposal_int_state = copy(self.current_integer_vector)
            new_proposal_state = self.gibbs_update(proposal_int_state, sub_dim)
            count = 0
            while np.any(new_proposal_state > self.int_lattice_bounds):
                count +=1
                new_proposal_state = self.gibbs_update(proposal_int_state, sub_dim)
            if af.metropolis_filter_simple(self.current_integer_vector, proposal_int_state,
                                           self.basis, self.latt_gauss_sigma):
                self.current_integer_vector = new_proposal_state
        self.markov_chain.append(copy(self.current_integer_vector))
        self.lattice_markov_chain.append(copy(self.basis.T @ self.current_integer_vector))

    def run(self):
        for i in range(self.markov_iters):
            self.update_state()


def multi_run(pars, it):
    """
        Running a multiprocessed version of the experiment.
    :param pars: parameters for the run, dict
    :param it: current iteration, int
    :return: writes results to file.
    """
    if it in os.listdir(pars['path']):
        return 1
    else:
        new_path = pars['path'] + '/' + str(it)
        os.mkdir(new_path)
        experiment = WalkExperiment(
            pars['basis'],
            pars['prob_density_2'],
            pars['prob_density_3'],
            pars['coords_2'],
            pars['coords_3'],
            pars['graph_bounds_2'],
            pars['graph_bounds_3'],
            pars['markov_iters'],
        )
        experiment.run()
        ints = np.array(experiment.markov_chain).astype(int)
        latts = np.array(experiment.lattice_markov_chain).astype(int)
        np.savetxt(
            new_path+'/ints.csv',
            ints,
            delimiter=',',
            fmt='%5.0f'
        )
        np.savetxt(
            new_path+'/latts.csv',
            latts,
            delimiter=',',
            fmt='%5.0f'
        )


# Run
if __name__== "__main__":
    path = 'results'
    if str(config.lattice_num) not in os.listdir(path):
        os.mkdir(path + '/' + str(config.lattice_num))

    spec_pars = {
        'basis': config.lattice_basis,
        'prob_density_2': config.dist_2,
        'prob_density_3': config.dist_3,
        'coords_2': config.coords_2,
        'coords_3': config.coords_3,
        'graph_bounds_2': config.graph_bounds_2,
        'graph_bounds_3': config.graph_bounds_3,
        'markov_iters': 100,
        'path': path + '/' + str(config.lattice_num)
    }

    pool = mp.Pool(config.cores)

    if len(os.listdir(spec_pars['path'])) == 0:
        iterables = range(config.number_of_runs)
    else:
        (_, results_names, _) = next(os.walk(spec_pars['path']))
        result_numbers = [int(findall(r'\d+', string)[0]) for string in results_names]
        iterables = filterfalse(lambda x: x in result_numbers, range(config.number_of_runs))

    [pool.apply(multi_run,
                args = (spec_pars,
                        i))
     for i in iterables]

    pool.close()
    pool.join()
