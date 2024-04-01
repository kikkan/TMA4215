import numpy as np
import matplotlib.pyplot as plt
import random as rnd
from datetime import datetime as dt
import pdb
from givenFncs import generate_data as gdata

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
    # return 1 / np.cosh(x)**2 # our
    return 1.0 - np.tanh(x) ** 2


def eta(x):  # max [0, 1]
    # return (1.0 + np.tanh(x / 2.0)) / 2.0
    return x


def deta(x):
    # return 1/(2 * np.cosh(x) + 2) # our
    # return 0.25 * (1.0 - np.tanh(x / 2.0) ** 2)
    return np.ones_like(x)


def J(theta):  # (6)
    return 0.5 * np.linalg.norm(theta) ** 2


def scale(y, alpha, beta):
    b = np.max(y)
    a = np.min(y)
    return ((b - y)*alpha + (y - a)*beta) / (b-a)


class Network():
    def __init__(self, Y0, c, K=10, h=0.05, τ=0.1):
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
        self.wstart = np.random.rand(self.d, 1)
        self.μstart = rnd.random()
        # Set weigths for current run
        self.W = self.Wstart.copy()
        self.b = self.bstart.copy()
        self.w = self.wstart.copy()
        self.μ = self.μstart
        # Memory allocation
        self.Z = np.zeros((K+1, self.d, self.I))  # Y0 = Y0 ?
        self.Z[0] = Y0
        # One less than intended bc P[0] is not used
        self.P = np.zeros((K, self.d, self.I))
        self.dW = np.zeros((K, self.d, self.d))
        self.db = np.zeros((K, self.d, 1))
        self.dw = np.zeros_like(self.w)
        self.dμ = np.zeros_like(self.μ)
        self.Υ = np.zeros_like(self.c)
        self.ΥmC = self.Υ - self.c
        self.Jt = J(self.ΥmC)
        self.m = [np.zeros((self.K, self.d, self.d)),
                  np.zeros((self.K, self.d, 1)),
                  np.zeros((self.d, 1)),
                  0]  # m_0 for W_k, b_k, w_k and my
        self.v = [np.zeros((self.K, self.d, self.d)),
                  np.zeros((self.K, self.d, 1)),
                  np.zeros((self.d, 1)),
                  0]  # v_0 for W_k, b_k, w_k and my

    def setY0andC(self, Y0, c):
        self.Z = np.zeros((self.K+1, self.d, self.I))
        self.Z[0] = Y0
        self.P = np.zeros((self.K, self.d, self.I))
        self.c = c

    def dJdTheta(self):
        """computing Z_k from (4)"""
        for k in range(0, self.K):
            self.Z[k+1] = self.Z[k] + self.h * sigma(self.W[k]@self.Z[k] + self.b[k])  # (4)

        # to save computation time:
        ZcrossO = self.Z[-1].T@self.w + self.μ

        """computing P_K"""
        self.Υ = eta(ZcrossO).T  # (5)
        if np.linalg.norm(self.Υ) < 1:
            pass
        self.ΥmC = self.Υ - self.c
        # self.Jt = J(self.gamma - self.c)
        self.Jt = 0.5*np.linalg.norm(self.ΥmC)**2

        self.P[self.K-1] = np.outer(self.w, (self.ΥmC * deta(ZcrossO).T))  # (10) (verified)
        # self.P[self.K-1] = self.omega @ (self.Gmc * deta(ZcrossO).T) # Similar vitber

        """Backward propagation:"""
        # computing  P_k , k = [k-1, 1] from (11)
        # P is one size less than intended bc P[0] is not used, thus never computed.
        # for k in range(self.K - 1, 0, -1):
        #     self.P[k-1] = self.P[k] + self.h * np.dot(self.W[k].T,
        #                                      (dsigma(self.W[k] @ self.Z[k] + self.b[k]) * self.P[k]))
        for k in range(self.K - 1, 0, -1):
            self.P[k-1] = self.P[k] + self.h * (self.W[k].T @
                                                (dsigma(self.W[k] @ self.Z[k] + self.b[k]) * self.P[k]))

        # for k in range(self.K-1, 0, -1):
        #     self.P[k-1] = self.P[k] + self.h * np.transpose(self.W[k]) @ np.multiply(
        #         dsigma(self.W[k] @ self.W[k] + self.b[k]), self.P[k]).T

        """compute gradients (8) and (9)"""
        # self.dJdmu = deta(ZcrossO).T @ self.Gmc.T  # (8)
        self.dμ = np.sum(self.ΥmC.T * deta(ZcrossO))  # (their) same as our?
        # self.dJdomega = self.Z[self.K] @ (self.Gmc.T *
        #                                   deta(ZcrossO))  # (9)
        self.dw = self.Z[self.K] @ (self.ΥmC.T *
                                    deta(ZcrossO))  # (9)

        """calculating gradients (12) and (13)"""
        for k in range(0, self.K):
            # To save computation
            PhadWcrossZ = self.P[k] * dsigma(self.W[k] @ self.Z[k] + self.b[k])
            # Compute grads
            # self.dJdW[k] = self.h * np.dot(PhadWcrossZ, self.Z[k].T)
            # self.dJdb[k] = self.h * np.dot(PhadWcrossZ, np.ones((self.I, 1)))
            self.dW[k] = self.h * (PhadWcrossZ @ self.Z[k].T)
            self.db[k] = self.h * (PhadWcrossZ @ np.ones((self.I, 1)))

    def showEvolution(self):
        for k in range(0, self.K):
            self.Z[k + 1] = self.Z[k] + self.h * \
                sigma(self.W[k] @ self.Z[k] + self.b[k])

        for z in self.Z:
            eta(z.T@self.w + self.μ).T
            plt.plot(eta(z.T@self.w + self.μ).T[0])
            plt.plot(self.c[0])
            plt.show()

    def restart(self, K=None, h=None, τ=None):
        """Resets the weights

        Args:
            K (int, optional): Layers. Defaults to None.
            h (float, optional): Step size. Defaults to None.
            τ (float, optional): Learning param. Defaults to None.
        """
        self.W = self.Wstart.copy()
        self.b = self.bstart.copy()
        self.w = self.wstart.copy()
        self.μ = self.μstart.copy()


class PVGD(Network):
    """Plain Vanilla Gradient Descent

    Args:
        Network (class): super class
    """

    def __init__(self, Y0, c, K=10, h=0.05, τ=0.1, plotFreq=10000, logFreq=100):
        super().__init__(Y0, c, K, h, τ)
        self.log = {(K, h, τ): []}
        self.plotFreq = plotFreq
        self.j = 0

    def plainVanilla(self):
        self.W -= self.τ * self.dW
        self.b -= self.τ * self.db
        self.w -= self.τ * self.dw
        self.μ -= self.τ * self.dμ

    def run(self, tol, tot, h=0.05, τ=0.1):
        self.τ = τ
        self.h = h
        while self.j <= tot:
            self.dJdTheta()
            self.plainVanilla()

            self.j += 1
            # Check progression (deleteable)
            if (self.j % self.plotFreq == 0) and (self.j > tot*0.7):
                temp_x = np.linspace(0, 1, len(self.Υ[0]))
                plt.title('run {} of {} with params [{},{},{}]'.format(
                    self.j, int(tot), self.K, self.h, self.τ))
                plt.plot(temp_x, self.Υ[0], label='Gamma')
                plt.plot(temp_x, self.c[0])
                plt.legend()
                plt.show()

            if self.Jt < tol:
                print('Tolerance reached in {} iterations. (Jt = {})'.format(
                    self.j, self.Jt))
                break


class Adam(Network):  # Inherits from Network
    """The Adam method

    Args:
        Network (class): Super class
    """

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
            ndarray: Updated derivative of weight (gj)
        """
        b1 = 0.9
        b2 = 0.999
        a = 0.01
        epsilon = 1e-8
        mj = b1*mPrev + (1-b1)*gj
        vj = b2*vPrev + (1-b2)*(np.square(gj))
        mHatt = mj / (1-b1**self.j)
        vHatt = vj / (1-b2**self.j)
        thetaPrev -= a*mHatt / (np.sqrt(vHatt) + epsilon)
        return thetaPrev, mj, vj

    def run(self, tol, tot, h=0.05):
        self.h = h
        while self.Jt > tol and self.j <= tot:
            self.j += 1

            self.dJdTheta()  # Update derivatives of the weights

            self.W, self.m[0], self.v[0] = self.adam(
                self.dW, self.W, self.m[0], self.v[0])
            self.b, self.m[1], self.v[1] = self.adam(
                self.db, self.b, self.m[1], self.v[1])
            self.omega, self.m[2], self.v[2] = self.adam(
                self.dw, self.omega, self.m[2], self.v[2])
            self.mu, self.m[3], self.v[3] = self.adam(
                self.dμ, self.mu, self.m[3], self.v[3])

            self.Jt = J(self.Υ - self.c)
            # Check progression (deleteable)

            if self.j % self.plotFreq == 0:
                temp_x = np.linspace(0, 1, len(self.Υ[0]))
                plt.title('run {}'.format(self.j))
                plt.plot(temp_x, self.Υ[0], label='Gamma')
                plt.plot(temp_x, self.c[0], ls="--")
                plt.legend()
                plt.show()


class SGD():
    """Stochastic Gradient Descent
    """

    def __init__(self, method, Y0, c):
        self.method = method
        self.Y0 = Y0
        self.c = c
        self.sections = {}

    def sep(self, parts=0, length=0):
        if (parts == 0 and length == 0) or (parts != 0 and length != 0):
            raise ValueError
        elif parts:
            indexes = [i for i in range(len(self.c[0]))]  # selectable indexes
            for i in range(parts):
                self.sections[i] = []  # stores (input_i, c_i)

            j = 0
            while indexes:
                """Not done"""
                i = np.random.randint(0, len(indexes))
                self.sections[j % parts].append([self.Y0[i], self.c[0, i]])


"""Run with ex fncs"""
np.random.seed()
x = np.array([np.linspace(-2, 2, 50)])
c = scale(Fy2(x), 0, 1)
xScaled = scale(x, 0, 1)

# test2 = Adam(xScaled, c, plotFreq=20000)
# test2.run(1e-4, 8*1e4, h=0.05)

# ad = Adam(xScaled, c, plotFreq=int(1*1e4))
# ad.run(1e-4, 1*1e5, h=0.1)

pv = PVGD(xScaled, c, plotFreq=int(1*1e4))
pv.run(1e-4, 6*1e4, h=0.1, τ=0.01)

"""Run with data batches"""
# batch0 = gdata(0)
# x = scale(batch0['P'], 0, 1)

# c_v = scale(np.array([batch0['V']]), 0, 1)

# pv = PVGD(x, c_v, plotFreq=int(1e3/10))
# pv.run(1e-2, 1e3)
