# Moving the Graph Method
# Author: Edmund Dable-Heath
"""
    In this version of the quantum walk algorithm instead of having a fixed graph and the quantum walk taking steps
    around it the quantum walk will always walk from the centre of the graph with the graph being embedded into Z^n
    around the current point. This will allow for a much computationally simpler approach only requiring a single row of
    the propagator to be stored for the most part. We know the bounds on Z^n that will contain the shortest vector for a
    particular lattice so if the graph around a new point is going to step over the bounds we walk over a temporary
    reduced graph that is restricted by the boundary.
"""

import numpy as np
import auxillary_functions as af
from copy import copy
import math
import os
import multiprocessing as mp
import moving_walk_config as config
from re import findall
from itertools import filterfalse


class WalkExperiment:

    """
        Class for the walk experiment.

        Cost function has two choices:
            - 'gauss' selects the lattice gaussian as the target distribution.
            - 'log' selects |log(||Bx||)| as the cost function.
    """

    def __init__(
            self, lattice_basis, prop_density, graph_coords, graph_bounds, markov_iters, markov_comm,
            cost_choice='gauss', latt_gauss_sigma=None
    ):
        self.basis = lattice_basis
        self.dimension = self.basis.shape[0]
        self.int_lattice_bounds = self.dimension * np.log2(self.dimension) + np.log2(np.linalg.det(self.basis))

        self.prob_density = prop_density
        self.graph_coords = graph_coords
        self.graph_bounds = graph_bounds

        if latt_gauss_sigma is not None:
            self.latt_gauss_sigma = latt_gauss_sigma
        else:
            self.latt_gauss_sigma = math.sqrt(self.dimension) * \
                                    np.abs(np.linalg.det(self.basis))**(1/float(self.dimension))

        self.current_integer_vector = np.random.randint(-self.graph_bounds, self.graph_bounds+1, self.dimension)
        self.markov_chain = [copy(self.current_integer_vector)]
        self.lattice_markov_chain = [self.basis.T @ copy(self.current_integer_vector)]
        self.markov_iters = markov_iters
        self.markov_comm = markov_comm
        self.markov_cost_choice = cost_choice

    def update_state(self):
        proposal_int_state = self.current_integer_vector + \
                             np.asarray(self.graph_coords[np.random.choice([i
                                                                            for i in range(self.prob_density.shape[0])],
                                                                           p=self.prob_density.tolist())])
        while np.any(proposal_int_state > self.int_lattice_bounds):
            # print(proposal_int_state)
            proposal_int_state = self.current_integer_vector + \
                                 np.asarray(self.graph_coords[np.random.choice([i for i
                                                                                in range(self.prob_density.shape[0])],
                                                                               p=self.prob_density.tolist())])
        if self.markov_cost_choice == 'gauss':
            if af.metropolis_filter_simple(self.current_integer_vector,
                                           proposal_int_state, self.basis, self.latt_gauss_sigma):
                self.current_integer_vector = proposal_int_state
        elif self.markov_cost_choice == 'log':
            if af.metropolis_filter_log_cost(self.current_integer_vector, proposal_int_state, self.basis):
                self.current_integer_vector = proposal_int_state
        self.markov_chain.append(copy(self.current_integer_vector))
        self.lattice_markov_chain.append(self.basis.T @ copy(self.current_integer_vector))

    def run(self):
        for i in range(self.markov_iters):
            self.update_state()
            # print(self.markov_chain)


def multi_run(pars, it):
    """
        Running a multi-processed version of the experiment.
    :param pars: parameters for the run
    :param it: current iteration
    :return: writes results to file
    """
    if str(it) in os.listdir(pars['path']):
        return 1
    else:
        new_path = pars['path'] + '/' + str(it)
        os.mkdir(new_path)
        experiment = WalkExperiment(
            pars['basis'],
            pars['prob_density'],
            pars['coords'],
            pars['graph_bounds'],
            pars['markov_iters'],
            pars['markov_comm'],
            pars['cost_choice']
        )
        experiment.run()
        np.savetxt(new_path+'/ints.csv', X=np.array(experiment.markov_chain), delimiter=',')
        np.savetxt(new_path+'/latts.csv', X=np.array(experiment.lattice_markov_chain), delimiter=',')


# Run
if __name__ == "__main__":
    path = '/rds/general/user/ead17/ephemeral/ctqw_results/'+ str(config.gamma_mark) + '/' \
           + str(config.dimension) + '/' + config.lattice_type
    if str(config.lattice_num) not in os.listdir(path):
        os.mkdir(path + '/' + str(config.lattice_num))

    spec_pars = {
        'basis': config.lattice_basis,
        'prob_density': config.dist,
        'coords': config.coords,
        'graph_bounds': config.graph_bounds,
        'markov_iters': 10000,
        'markov_comm': 1e-7,
        'cost_choice': 'gauss',
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
                args=(spec_pars,
                      i))
     for i in iterables]

    pool.close()
    pool.join()
