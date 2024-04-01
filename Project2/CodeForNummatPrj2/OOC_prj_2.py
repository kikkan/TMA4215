import numpy as np
import matplotlib.pyplot as plt
import random as rnd
from datetime import datetime as dt
import pdb

np.seterr(all='raise')
# Example fncs


def Fy2(y):
    return 0.5*y**2


def Fcos(y):
    return 1 - np.cos(y)


def Fy22(y1, y2):
    return 0.5*(y1**2 + y2**2)


def Fsqrt(y1, y2):
    return - 1 / np.sqrt(y1**2 + y2**2)

##########################################


def sigma(x):
    return np.tanh(x)


def dsigma(x):
    return 1 / np.cosh(x)**2


def phi_k(y, h, Wk, bk):
    return y + h*sigma(Wk@y + bk)


def phi_K(y, w, mu):
    return eta(y.T @ w + mu)


def eta(x):  # max [0, 1]
    return (1 + np.tanh(x / 2)) / 2


def deta(x):
    return 1/(2 * np.cosh(x) + 2)


def J(theta):  # (6)
    return 0.5 * np.linalg.norm(theta) ** 2


def scale(y, alpha, beta):
    b = np.max(y)
    a = np.min(y)
    return ((b - y)*alpha + (y - a)*beta) / (b-a)


class Network():
    def __init__(self, Y0=np.array([]), c=np.array([]), K=10, h=0.05, τ=0.1):
        """[summary]

        Args:
            Y0 ([type], optional): [description]. Defaults to np.array([]).
            c ([type], optional): [description]. Defaults to np.array([]).
            K (int, optional): [description]. Defaults to 10.
            h (float, optional): [description]. Defaults to 0.05.
        """
        self.d = len(Y0)
        self.I = len(Y0[0])
        self.c = c
        self.K = K
        self.h = h
        self.τ = τ
        # Set start weights
        # Set start weights for comparison purposes
        self.Wstart = np.random.rand(K, self.d, self.d)
        self.bstart = np.random.rand(K, self.d, 1)
        self.omegastart = np.random.rand(self.d, 1)
        self.mustart = rnd.random()
        # Set weigths for current run
        self.W = self.Wstart.copy()
        self.b = self.bstart.copy()
        self.omega = self.omegastart.copy()
        self.mu = self.mustart
        # Memory allocation
        self.Z = np.zeros((K+1, self.d, self.I))  # Y0 = Y0 ?
        self.Z[0] = Y0
        # One less than intended bc P[0] is not used
        self.P = np.zeros((K, self.d, self.I))
        self.dJdW = np.zeros((K, self.d, self.d))
        self.dJdb = np.zeros((K, self.d, 1))
        self.dJdomega = np.zeros_like(self.omega)
        self.dJdmu = np.zeros_like(self.mu)
        self.gamma = np.zeros_like(self.c)
        self.Gmc = self.gamma - self.c
        self.Jt = J(self.Gmc)
        self.m = [np.zeros((self.d, self.d)), np.zeros((self.d, 1)), np.zeros(
            (self.d, 1)), 0]  # m_0 for W_k, b_k, w_k and my
        self.v = [np.zeros((self.d, self.d)), np.zeros((self.d, 1)), np.zeros
                  ((self.d, 1)), 0]  # v_0 for W_k, b_k, w_k and my

    # Basic functions

    def dJdTheta(self):
        # computing Z_k from (4)

        for k in range(0, self.K):
            self.Z[k+1] = self.Z[k] + self.h * \
                sigma(self.W[k]@self.Z[k] + self.b[k])  # (4)

        # computing P_K
        self.gamma = eta(self.Z[self.K].T@self.omega + self.mu).T  # (5)
        self.Gmc = self.gamma - self.c

        self.P[self.K-1] = np.outer(self.omega, ((self.gamma - self.c) *
                                                 deta(self.Z[self.K].T @ self.omega + self.mu).T))  # (10) (verified)

        # calculating gradients (8) and (9)
        self.dJdmu = deta(self.Z[self.K].T * self.omega +
                          self.mu).T @ (self.gamma - self.c).T  # (8)
        self.dJdomega = self.Z[self.K] @ ((self.gamma - self.c).T *
                                          deta(self.Z[self.K].T @ self.omega + self.mu))  # (9)

        # Backward propagation:
        # computing  P_k , k = [k-1, 1] from (11)
        # P is one size less than intended bc P[0] is not used.
        for k in range(self.K - 1, 0, -1):
            self.P[k-1] = self.P[k] + \
                np.dot(
                    self.h * self.W[k].T, (dsigma(self.W[k] @ self.Z[k] + self.b[k]) * self.P[k]))

        # calculating gradients (12) and (13)
        for k in range(0, self.K):
            self.dJdW[k] = self.h * \
                np.dot(self.P[k] * dsigma(self.W[k]
                                          @ self.Z[k] + self.b[k]), self.Z[k].T)
            self.dJdb[k] = self.h * np.dot(self.P[k] * dsigma(
                self.W[k] @ self.Z[k] + self.b[k]), np.ones((self.I, 1)))

    def plainVanilla(self):
        self.W -= self.τ * J(self.dJdW)
        self.b -= self.τ * J(self.dJdb)
        self.mu -= self.τ * J(self.dJdmu)
        self.omega -= self.τ * J(self.dJdomega)

    def runPlainVanilla(self, tol, tot, h=0.05, τ=0.1):
        j = 1
        while np.linalg.norm(self.Jt) > tol and j <= tot:
            self.dJdTheta()
            self.plainVanilla()
            self.Jt = J(self.gamma - self.c)
            j += 1
            # Check progression (deleteable)
            if j % 1000 == 0 or j == 2:
                temp_x = np.linspace(0, 1, len(self.gamma[0]))
                plt.plot(temp_x, self.gamma[0], label='Gamma')
                plt.plot(temp_x, self.c[0])
                plt.legend()
                plt.show()

    def showEvolution(self):
        for k in range(0, self.K):
            self.Z[k + 1] = self.Z[k] + self.h * \
                sigma(self.W[k] @ self.Z[k] + self.b[k])

        for z in self.Z:
            eta(z.T@self.omega + self.mu).T
            plt.plot(eta(z.T@self.omega + self.mu).T[0])
            plt.plot(self.c[0])
            plt.show()

    def restart(self, K=None, h=None, τ=None):
        """Resets the weights

        Args:
            K (int, optional): Layers. Defaults to self.K.
            h (float, optional): Step size. Defaults to self.h.
            τ (float, optional): Learning param. Defaults to self.τ.
        """
        self.W = self.Wstart.copy()
        self.b = self.bstart.copy()
        self.omega = self.omegastart.copy()
        self.mu = self.mustart.copy()


class Adam(Network):  # Inherits from Network
    def __init__(self, Y0, c, K=10, h=0.05, τ=0.1, plotFreq=10000, logFreq=100):
        super().__init__(Y0, c, K, h, τ)
        self.log = {(K, h, τ): []}
        self.plotFreq = plotFreq
        self.j = 0

    def adam(self, gj, thetaPrev, mPrev, vPrev):  # Should this be member fnc
        """The Adam Algorithm

        Args:
            gj (ndarray): Weight
            thetaPrev (ndarray): Previous derivative of weight (gj)
            mPrev (ndarray): Previous value of m
            vPrev (ndarray): Previous value of v
            j (int): Number of iterations

        Returns:
            ndarray: New derivative of weight (gj)
        """
        b1 = 0.9
        b2 = 0.999
        a = 0.01
        epsilon = 1E-8
        # pdb.set_trace()
        mj = b1*mPrev + (1-b1)*gj
        vj = b2*vPrev + (1-b2)*(gj*gj)
        mHatt = mj / (1-b1**self.j)
        vHatt = vj / (1-b2**self.j)
        thetaPrev -= a*mHatt / (np.sqrt(vHatt) + epsilon)
        return thetaPrev, mj, vj

    def run(self, tol, tot, h=0.05, tau=0.1):
        while self.Jt > tol and self.j <= tot:
            self.j += 1

            self.dJdTheta()  # Update derivatives of the weights

            self.W, self.m[0], self.v[0] = self.adam(
                self.dJdW, self.W, self.m[0], self.v[0])
            self.b, self.m[1], self.v[1] = self.adam(
                self.dJdb, self.b, self.m[1], self.v[1])
            self.omega, self.m[2], self.v[2] = self.adam(
                self.dJdomega, self.omega, self.m[2], self.v[2])
            self.mu, self.m[3], self.v[3] = self.adam(
                self.dJdmu, self.mu, self.m[3], self.v[3])

            self.Jt = J(self.gamma - self.c)
            # Check progression (deleteable)

            if self.j % self.plotFreq == 0:
                temp_x = np.linspace(0, 1, len(self.gamma[0]))
                plt.title('run {}'.format(self.j))
                plt.plot(temp_x, self.gamma[0], label='Gamma')
                plt.plot(temp_x, self.c[0], ls="--")
                plt.legend()
                plt.show()


x = np.array([np.linspace(-2, 2, 150)])
c = scale(Fy2(x), 0, 1)
xScaled = scale(x, 0, 1)

test = Network(xScaled, c, 20)

# test.runAdam(1e-4, 3*1e3)
# test.showEvolution()

# test.runPlainVanilla(1e-4, 2*1e3, τ=0.05)

test2 = Adam(xScaled, c, h=0.08, plotFreq=20000)
test2.run(1e-4, 8*1e4, h=0.08)
