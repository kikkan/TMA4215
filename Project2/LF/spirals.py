import math
import numpy as np

    
def get_data_spiral_2d(n_samples=200):
    '''Create "observations" for training.
    
    Creates n_samples dots on two intertwined spirals.
    The dots are classified based on which spiral they belong to and
    then fuzzed by perturbing the coordinates and permuting the ordering.
    
    Return values
    
    features: numpy array of shape (2, n_samples). Each column is the x and y coordinates of a point.
    labels: numpy aray of shape (n_samples, 1).
    Each entry is a bool, indicating if the point belongs to one group or the other.
    '''
    m1 = math.ceil(n_samples / 2)
    m2 = n_samples - m1
    
    n_turns = 1.0
    
    phi1 = np.pi
    d1 = _make_spiral(m1, phi1, n_turns)

    
    phi2 = (phi1 + np.pi) % (2.0 * np.pi)
    d2 = _make_spiral(m2, phi2, n_turns)
    
    features = np.hstack((d1, d2))
    labels = np.ones((n_samples,1), dtype='bool_')
    labels[m1:] = False
    
    features = features + .05 * np.random.randn(*features.shape)
    
    indexes = np.random.permutation(n_samples)
    features = features[:, indexes]
    labels = labels[indexes]


    return features, labels


def _make_spiral(m, phi, n_turns):
    '''Makes points on a spiral
    
    This is a utility function for get_data_spiral_2d'''
    r = np.linspace(0.1, 1.0, m)
    a = np.linspace(0.1, 2.0 * np.pi * n_turns, m)
    xs = r * np.cos(a + phi)
    ys = r * np.sin(a + phi)
    return np.stack([xs, ys])
