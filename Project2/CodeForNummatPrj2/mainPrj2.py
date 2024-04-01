import numpy as np
import random as rnd
from datetime import datetime as dt

# Example functions


def Fy2(y):
    return 0.5*y**2


def Fcos(y):
    return 1 - np.cos(y)


def Fy22(y1, y2):
    return 0.5*(y1**2 + y2**2)


def Fsqrt(y1, y2):
    return - 1 / np.sqrt(y1**2 + y2**2)


"""Fra willirom"""
# Simple fncs


def sigma(x):
    return np.tanh(x)


def dsigma(x):
    return x  # regn ut derivert


def phi_k(y, h, Wk, bk):
    return y + h*sigma(Wk*y + bk)


def phi_K(y, w, my):
    return eta(y.T * w + my)


def eta(x):
    return (1 + np.tanh(x/2))/2


def deta(x):
    return 1/(2 * np.cosh(x) + 2)


def Adam(gj, thetaPrev, mPrev, vPrev, j):
    b1 = 0.9
    b2 = 0.999
    a = 0.01
    epsilon = 1E-8
    mj = b1*mPrev + (1-b1)*gj
    vj = b2*vPrev + (1-b2)*(gj*gj)
    mHatt = mj / (1-b1**j)
    vHatt = vj / (1-b2**j)
    thetaj = thetaPrev - a*mHatt / (np.sqrt(vHatt) + epsilon)
    return thetaj, mj, vj


def algorithm(K, tau, Y0, W, b, omega, h, my, d, c):  # felles notasjon må på plass
    """
        Args:
            K (Int): [layers]
            tau (float): learning paramter
            Y0 (d x I ?): start values
            W (K * (d x d)): array of (d x d) weights
            b (k x d): array of d-dim vectors
            omega (d-dim vector): final layer weight
            my (float): final layer weight
            h (float): step lenght
            mj (float?) : Adam start
            vj (float?) : Adam start
            j (int) : iter index
    """
    c = Fy2(Y0).T  # er dette noe vi får fra hver batch? Ja
    # Memory allocation
    Z = np.zeros((K, d, I))
    P = np.zeros((K, d, I))
    dJdW = np.zeros((K, d, d))
    dJdb = np.zeros((K, d, 1))
    mj = 0  # disse to skal kasnkje være arrays? Believe so! se Adam descent algo kor m0 og v0 blir definert på side 9
    vj = 0
    j = 0
    Z[2] = Y0
    # Time the code
    start = dt.now()
    Gamma = np.zeros_like(c)
    while True:

        # computing Z_k from (4)
        for k in range(0, K-1):
            Z_next = phi_k(Z[k], h, W[k], b[k])
            Z[k+1] = Z_next

        # computing P_K
        Gamma = eta(Z[K-1].T*omega + my)
        P[K-1] = np.outer(omega, (Gamma - c) *
                          deta((Z[K-1]).T * omega + my))

        # calculating gradients (8) and (9)
        dJdmy = deta(Z[K-1].T * omega + my * np.ones((d, 1))).T * (Gamma - c)
        dJdomega = Z[K-1] * ((Gamma - c) * deta((Z[K-1]).T * omega + my))

        # computing  P_k , k = [k-1, 1] from (11)
        for k in range(K - 2, 0, -1):
            P[k] = P[k+1] + \
                np.dot(h * W[k].T, (dsigma(W[k] * Z[k] + b[k]) * P[k]))

        # calculating gradients (8) and (9)

        for k in range(0, K-1):
            dJdW[k] = h * np.dot(P[k+1] * dsigma(W[k]*Z[K-1] + b[k]), Z[k].T)
            # temp_PWZ = h * np.dot(P[k+1] * dsigma(W[k]*Z[K-1] + b[k])
            dJdb[k] = h * \
                np.dot(P[k+1] * dsigma(W[k]*Z[K-1] + b[k]), np.ones((I, 1)))
            # dJdb[k] = h * np.dot(P[k+1] * dsigma(W[k]*Z[K-1] + b[k]), np.ones((d, 1)))

        # updating weights using adam
        newW = Adam(dJdW, W, mj, vj, j)
        newb = Adam(dJdb, b, mj, vj, j)
        newomega = Adam(dJdomega, omega, mj, vj, j)
        newmy = Adam(dJdmy, my, mj, vj, j)
        j += 1
        if j % 1000 == 0:
            print('Now on run {}.\nTime elapsed {}'.format(j, dt.now() - start))
        if np.linalg.norm(Gamma - c, 2)**2 / 2 < 1e-6:
            break
    return newW, newb, newomega, newmy


# oppretter tilfeldige parametere
Y0 = np.array([np.linspace(-2, 2, 250)])
c = Fy2(Y0)
K = 20
d = len(Y0)  # hvordan er input?'
W = np.random.rand(K, d, d)
b = np.random.rand(K, d, 1)
omega = np.random.rand(d, 1)
mu = rnd.random()
h = 0.01
I = len(Y0[0])  # kan velge antall punkter som vi valgte bilder i vitber?


# find the best weights
newW, newb, newomega, newmy = algorithm(K, 0.1, Y0, W, b, omega, h, mu, d, c)
