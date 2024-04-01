import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt

"""This file contain..."""


"""##################### Example fncs"""


def F2(y):
    return 0.5*y**2


def Fcos(y):
    return 1 - np.cos(y)


def F22(y1, y2):
    return 0.5*(y1**2 + y2**2)


def Fsqrt(y1, y2):
    return - 1 / np.sqrt(y1**2 + y2**2)


"""###################### Simple fncs"""


def σ(x):
    return np.tanh(x)


def dσ(x):
    # return 1 / np.cosh(x)**2 # our
    return 1.0 - np.tanh(x) ** 2


def η(x):  # max [0, 1]
    return (1.0 + np.tanh(x / 2.0)) / 2.0


def dη(x):
    # return 1/(2 * np.cosh(x) + 2) # our
    return 0.25 * (1.0 - np.tanh(x / 2.0) ** 2)


def scale(y, α=0, β=1):
    b = np.max(y)
    a = np.min(y)
    return ((b - y)*α + (y - a)*β) / (b-a)


def scaleBack(sy, c, α=0, β=1):
    b = np.max(c)
    a = np.min(c)
    return sy*(b - a)/(β - α) - (b*α - a*β)/(β-α)


def scaleBackGrad(c, α, β):
    a = np.min(c)
    b = np.max(c)
    return (b-a) / (β - α)


"""UI fncs"""


def plotApprox(C, *args, x=np.array([])):
    """Plots optimzers best approximation to one set of training data

    Args:
        C       (ndarray)          : The exact value
        *args   (tuple)            : tuple of optimized models that should be compared
        x       (ndarray, optional): The x-axis of the training data. Defaults to np.array([]).
    """
    if len(x) == 0:
        x = np.linspace(0, 1, args[0].M.I)
    for mod in args:
        lbl = mod.name()
        if lbl == 'PVGD':
            lbl += ', h={}, τ={}, J(θ)={}'.format(
                mod.M.h, mod.τ, round(mod.M.Jθ, 3))
        else:
            lbl += ', h={}, J(θ)={}'.format(mod.M.h, round(mod.M.Jθ, 3))

        plt.plot(x, scaleBack(mod.M.Υ[0], C), label=lbl, linewidth=3)

    if C.shape[0] == 1:
        plt.plot(x, C[0], label='F(y)', ls='--')
    else:
        plt.plot(x, C, label='F(y)', ls='--')
    plt.title(r'Comparing $\tilde{F}(y)$ with $F(y)$')
    plt.legend()
    plt.show()
