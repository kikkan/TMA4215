'''
This module contains three functions that can plot some aspects of the simple 2d-points-in-a-spiral problem.
The functions can be used as is, without looking into their inner workings.
You can also use them as inspiration for other ways of visualizing what your neural net does.
'''
import numpy as np
import matplotlib.pyplot as plt


def plot_progression(Y, C):
    '''
    This function should make K+1 plots showing how, as the network progresses through its layers,
    the datapoints are drawn apart in such a way that they can be separated by a straight line.
    
    Y: A K+1-by-I-by-2 matrix of the values for each data point in the training set in each layer.
    C: An I-by-1 matrix of the labels corresponding to the datra points in Y.
    '''
    for k in range(Y.shape[0]):
        show_dots(Y[k,:,:],C.flatten())
        plt.show()


def plot_model(forward_function, Ys, C, n):
    '''
    Make a map that shows what part of the 2d plane is classified as belonging to which spiral arm.
    Also plot the training data in Ys, C.
    
    forward_function: A function that takes one argument, an S-by-2 matrix of S datapoints, and
        returns a vector of S classification values.
        
        Hint: This function will use the weights you have found, so you might want it to be a method on a
        class called something like Network or Model.
    Ys: An I-by-2 matrix. Corresponding to Y[0,:,:]
    C: An I-by-1 matrix of the labels corresponding to the datra points in Ys.
    n: Number of test-points in each direction. This controls the resolution of the plot.
    '''
    grid, coordinates = get_discretization(Ys, n)

    Z = forward_function(grid)
    l = np.linspace(0,1,8)
    l = np.array([shading(x) for x in l])

    plot_contours(*coordinates, Z, l, Ys, C.flatten())


def plot_separation(last_function, Ys, C, n):
    '''
    Show how the training data is represented in the last layer. Also maps the rest of the possible points in the plane.
    
    last_function: A function that takes one argument, and S-by-2 matrix of S intermediate states in
        the network, and retruns a vector of S classification values.
        It should multiply by w, add μ and evaluate 𝜂.
        
        Hint: This function will use the weights you have found, so you might want it to be a method on a
        class called something like Network or Model.
    Ys:An I-by-2 matrix. Corresponding to Y[-1,:,:].
    C: An I-by-1 matrix of the labels corresponding to the datra points in Ys.
    n: Number of test-points in each direction. This controls the resolution of the plot.
    '''
    grid, coordinates = get_discretization(Ys, n)

    Z = last_function(grid)
    l = np.linspace(0,1,500)

    plot_contours(*coordinates, Z, l, Ys, C.flatten())


######## Internals


def show_dots(positions, labels):
    '''Visualize the output of get_data_spiral_2d'''
    plt.scatter(x=positions[0,:], y=positions[1,:], s=1, c=labels, cmap='bwr')
    plt.axis([-1.2, 1.2, -1.2, 1.2])
    plt.axis('square')


def plot_contours(box, xcoordinates, ycoordinates, Z, l, Ys, C1):
    n = xcoordinates.size
    plt.contourf(xcoordinates, ycoordinates, Z.reshape((n,n)), cmap='seismic', levels=l)
    plt.contour(xcoordinates, ycoordinates, Z.reshape((n,n)), levels=1, colors='k')
    plt.scatter(x=Ys[0,:], y=Ys[1,:], s=1, c=C1, cmap='bwr')
    plt.axis(box)
    plt.axis('equal')
    plt.show()


def get_discretization(Ys, n):
    xmin, xmax, ymin, ymax = get_box(Ys)
    xcoordinates = np.linspace(xmin, xmax, n)
    ycoordinates = np.linspace(ymin, ymax, n)
    grid = get_grid(xcoordinates, ycoordinates)
    coordinates = ([xmin, xmax, ymin, ymax], xcoordinates, ycoordinates)
    return grid, coordinates


def get_box(Ys):
    xmin = min(Ys[0,:])
    xmax = max(Ys[0,:])
    xdelta = xmax-xmin
    xmin -= 0.2*xdelta
    xmax += 0.2*xdelta
    ymin = min(Ys[1,:])
    ymax = max(Ys[1,:])
    ydelta = ymax-ymin
    ymin -= 0.2*ydelta
    ymax += 0.2*ydelta
    return xmin, xmax, ymin, ymax

    
def get_grid(xcoordinates, ycoordinates):
    xv, yv = np.meshgrid(xcoordinates, ycoordinates)
    xs = xv.reshape(-1)
    ys= yv.reshape(-1)
    grid = np.stack([xs,ys])
    return grid


def shading(x):
    return shading1(shading2(x))


def shading1(x):
    if x == 0.0:
        return 0.0
    return 0.5 * np.tanh(np.tan(x * np.pi + np.pi / 2.0)) + 0.5


def shading2(x):
    if x < 0.5:
        return 0.5 - np.sqrt(0.25 - x**2)
    else:
        return 0.5 + np.sqrt(0.25 -(x-1.0)**2)
