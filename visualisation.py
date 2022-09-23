# Visualising the DHMC algorithm
# Authour: Edmund Dable-Heath
# Creating visualisations for the various aspects of the DHMC algorithm, these come in two different forms:
# 1. Visualising the space, i.e. the discontinuous 'distribution' over the integers. This will be done in both 2D and 3D
# 2. Plotting the path of a run of the algorithm, potentially animating it. Likely just in 2D but if it looks good in 3D
# then there is no reason not to if it is fairly straightforward to do.
#
# NB: Dimensions here refer to the dimension of visualisation, for the time being all visualisation are of two
# dimensional problems.

# import statements
import numpy as np
import matplotlib.pyplot as plt
import auxillary_functions as fn
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.cm as cm
import matplotlib
from mpl_toolkits.mplot3d import Axes3D


def two_dimensional_integer_tiling(basis, shortest_vector):
    """
    creating the two dimensional tiling of the integers with coloured voronoi cells.
    :param basis: lattice basis, entirely defines the scope of the entire problem. Can be a good or bad basis.
    :param shortest_vector: shortest_vector in the lattice
    :return: plot of two dimensional plane tiling
    """
    # Find range of integers needed
    int_range = fn.range_calc(basis)

    # Find points to be plotted
    int_x, int_y = np.mgrid[-int_range:int_range, -int_range:int_range]
    integer_points = np.c_[int_x.ravel(), int_y.ravel()]

    # Finding the colouring from the vector norms
    zero_point_energy = np.linalg.norm(np.dot(basis.T, np.array([int_range, int_range])))
    z = np.zeros(len(integer_points))
    for i in range(len(z)):
        z[i] = fn.potential_energy(basis, integer_points[i], zero_point_energy)
    minima = min(z)
    maxima = max(z)
    norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.viridis)

    # Set up voronoi object
    vor = Voronoi(integer_points)

    # plot voronoi cells
    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax1 = voronoi_plot_2d(vor, show_points=False, show_vertices=False, s=1)
    for r in range(len(vor.point_region)):
        region = vor.regions[vor.point_region[r]]
        if not -1 in region:
            polygon = [vor.vertices[i] for i in region]
            plt.fill(*zip(*polygon), color=mapper.to_rgba(z[r]))

    # plotting shortest vector as well
    shortest_ints = np.linalg.solve(basis.T, shortest_vector)
    ax2 = fig.add_subplot()
    ax2.scatter(shortest_ints[0], shortest_ints[1])
    ax2.scatter(-shortest_ints[0], -shortest_ints[1])

    plt.show()


def giants_causeway_graph(basis, shortest_vector):
    """
    plotting a 3d representation of the discontinous space for the two dimensional problem. also known as a LEGO plot
    :param basis: lattice basis, which full expalins the whole problem in this instance.
    :param shortest_vector: shortest vector in lattice, for plotting that on the visualisation.
    :return: plot of the 3d representation, similar looking to the giants causeway
    """
    # setup figure and axes
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')

    # Find range of integers needed
    int_range = 15

    # Find points to be plotted
    int_x, int_y = np.mgrid[-int_range:int_range, -int_range:int_range]
    # print(int_x)
    integer_points = np.c_[int_x.ravel(), int_y.ravel()]
    x, y = int_x.ravel(), int_y.ravel()

    # bar plot dimensions
    # zero_point_energy = np.linalg.norm(np.dot(basis.T, np.array([int_range, int_range])))
    width = depth = 1
    z = np.zeros_like(x)
    for i in range(len(z)):
        z[i] = fn.potential_energy(basis, integer_points[i], 0)
    bottom = 3*(z/4)

    # colouring
    minima = min(z)
    maxima = max(x)
    cmap = cm.get_cmap('viridis')
    rgba = [cmap((k-minima) / maxima) for k in z]

    # shortest vector algebraic solution
    shortest_vector = np.array([1, 0])
    shortest_ints = np.linalg.solve(basis.T, shortest_vector)

    ax.bar3d(x-0.5, y-0.5, bottom, width, depth, z, shade=True, color=rgba)
    # ax.plot3D([-shortest_ints[0], -shortest_ints[0]],
    #           [-shortest_ints[1], -shortest_ints[1]],
    #           [0, np.max(z)], 'gray')
    # ax.scatter(-shortest_ints[0], -shortest_ints[1], np.max(z))
    # ax.plot3D([shortest_ints[0], shortest_ints[0]],
    #           [shortest_ints[1], shortest_ints[1]],
    #           [0, np.max(z)], 'gray')
    # ax.scatter(shortest_ints[0], shortest_ints[1], np.max(z))
    plt.show()


# testing
if __name__ == "__main__":
    for i in range(32):
        b = np.genfromtxt(f'Lattices/2/{str(i)}_1.csv', delimiter=',', dtype=None)
        print(b)
        giants_causeway_graph(b, 0)
        input('press enter to continue...')



