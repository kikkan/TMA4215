import os
import struct
from array import array as pyarray
import numpy as np


def get_dataset(dataset="training", digit1=2, digit2=7, path="."):
    images, labels = load_mnist(dataset=dataset, path=path)
    images = np.reshape(images, newshape=(images.shape[0],-1), order='C') # Create (60 000 x whatever) 2D-matrix.
    labels = np.array(labels)

    label_indexes = (labels == digit1) | (labels == digit2)
    labels = labels[label_indexes]
    images = images[label_indexes]
    return images.T, (labels == digit1).reshape((-1,1))


### Internals:


def load_mnist(dataset="training", digits=np.arange(10), path="."):
    fname_img, fname_lbl = get_file_names(dataset, path)
    lbl = read_labels(fname_lbl)
    img, size, rows, cols = read_images(fname_img)

    images = np.array(img, order='C', dtype=np.uint8)
    images = images.reshape(size, rows, cols)
    labels = np.zeros((size, 1), dtype=np.int8)
    for i in range(size):
        labels[i] = lbl[i]
    labels2 = [label[0] for label in labels]
    return images, labels2


def get_file_names(dataset, path):
    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")
    return fname_img, fname_lbl


def read_labels(fname_lbl):
    with open(fname_lbl, 'rb') as flbl:
        _, size = struct.unpack(">II", flbl.read(8))
        lbl = pyarray("b", flbl.read())
    return lbl


def read_images(fname_img):
    with open(fname_img, 'rb') as fimg:
        _, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = pyarray("B", fimg.read())
    return img, size, rows, cols
