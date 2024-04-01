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

# Simple fncs


def σ(x):
    return np.tanh(x)


def dσ(x):
    # return 1 / np.cosh(x)**2 # our
    return 1.0 - np.tanh(x) ** 2


def η(x):  # max [0, 1]
    return (1.0 + np.tanh(x / 2.0)) / 2.0
    # return x # Identity fnc


def dη(x):
    return 0.25 * (1.0 - np.tanh(x / 2.0) ** 2)
    # return np.ones_like(x)


def scale(y, alpha, beta):
    b = np.max(y)
    a = np.min(y)
    return ((b - y)*alpha + (y - a)*beta) / (b-a)


################################# Misc fncs ###########################


def plotResult(model):
    app = model.Υ
    c = model.C
    plt.plot(app[0])
    plt.plot(c[0])
    plt.show()


def plotJevolution(model):
    pass

# Classes


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
        self.j = 0
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
        self.JΘ = 0.5 * np.linalg.norm(self.ΥmC)
        self.m = [np.zeros((self.K, self.d, self.d)),
                  np.zeros((self.K, self.d, 1)),
                  np.zeros((self.d, 1)),
                  0]  # m_0 for W_k, b_k, w_k and my
        self.v = [np.zeros((self.K, self.d, self.d)),
                  np.zeros((self.K, self.d, 1)),
                  np.zeros((self.d, 1)),
                  0]  # v_0 for W_k, b_k, w_k and my

    def setY0andC(self, Y0, c):
        self.I = len(c)
        self.Z = np.zeros((self.K+1, self.d, self.I))
        self.Z[0] = Y0
        self.P = np.zeros((self.K, self.d, self.I))
        self.c = c

    def computeZ(self, W, b, h, K):  # (4)
        for k in range(K):
            self.Z[k+1] = self.Z[k] + h * σ(W[k] @ self.Z[k] + b[k])

    def backwardPropagation(self, W, b, w, μ, Z, h, K):
        # Save computation
        Zxw = Z[-1].T @ w + μ
        # Update variables (Approximation)
        self.Υ = η(Zxw).T  # (5)
        self.ΥmC = self.Υ - self.c
        self.JΘ = 0.5*np.linalg.norm(self.ΥmC)**2
        # Compute last P[K]
        self.P[self.K-1] = np.outer(w, (self.ΥmC * dη(Zxw).T))  # (10) (verified)
        # self.P[self.K-1] = self.omega @ (self.Gmc * dη(Zxw).T) # Similar vitber
        """Backward propagation"""
        for k in range(K-1, 0, -1):  # Change P index to K+1
            self.P[k-1] = self.P[k] + h*W[k].T @ (dσ(W[k] @ Z[k] + b[k]) * self.P[k])  # (11)

    def computeGradients(self, W, b, w, μ, Z, P, h, K):
        # save computation
        dηZw = dη(Z[K].T @ w + μ)
        # compute (8) and (9)
        self.dμ = dηZw.T @ self.ΥmC.T
        self.dw = Z[K] @ (self.ΥmC.T * dηZw)
        # Compute (12) and (13)
        for k in range(K):
            # Save computation
            PhadσWZ = h * P[k] * dσ(W[k] @ Z[k] + b[k])
            self.dW[k] = PhadσWZ @ Z[k].T
            self.db[k] = PhadσWZ @ np.ones((self.I, 1))

    def update(self):
        self.computeZ(self.W, self.b, self.h, self.K)
        self.backwardPropagation(self.W, self.b, self.w, self.μ, self.Z, self.h, self.K)
        self.computeGradients(self.W, self.b, self.w, self.μ, self.Z, self.P, self.h, self.K)

    def logic(self):
        Z = np.zeros_like(self.Z)
        for k in range(0, self.K):
            Z[k + 1] = self.Z[k] + self.h * σ(self.W[k] @ self.Z[k] + self.b[k])
        l = 0
        for z in Z:
            plt.figure()
            plt.title(f'Layer {l}')
            η(z.T@self.w + self.μ).T
            plt.plot(η(z.T@self.w + self.μ).T[0])
            plt.plot(self.c[0], ls='--')
            l += 1
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
        self.μ = self.μstart

        self.j = 0

        self.Υ = np.zeros_like(self.c)
        self.ΥmC = self.Υ - self.c
        self.JΘ = 0.5 * np.linalg.norm(self.ΥmC)
        self.m = [np.zeros((self.K, self.d, self.d)),
                  np.zeros((self.K, self.d, 1)),
                  np.zeros((self.d, 1)),
                  0]  # m_0 for W_k, b_k, w_k and my
        self.v = [np.zeros((self.K, self.d, self.d)),
                  np.zeros((self.K, self.d, 1)),
                  np.zeros((self.d, 1)),
                  0]  # v_0 for W_k, b_k, w_k and my
        if h:
            self.h = h
        if τ:
            self.τ = τ


class PVGD(Network):
    """Plain Vanilla Gradient Descent

    Args:
        Network (class): super class
    """

    def __init__(self, Y0, c, K=10, h=0.05, τ=0.1, plotFreq=10000, logFreq=100, plotProg=False):
        super().__init__(Y0, c, K, h, τ)
        self.log = {(K, h, τ): []}
        self.plotFreq = plotFreq
        self.plotProg = plotProg
        self.logFreq = logFreq
        self.JθLog = []

    def plainVanilla(self):
        self.W -= self.τ * self.dW
        self.b -= self.τ * self.db
        self.w -= self.τ * self.dw
        self.μ -= self.τ * self.dμ

    def run(self, tol, tot, h=0.05, τ=0.1):
        self.τ = τ
        self.h = h
        while self.j <= tot:
            self.update()
            self.plainVanilla()

            self.j += 1
            # Check progression (deleteable)
            if self.plotProg:
                if (self.j % self.plotFreq == 0) and (self.j > tot*0.8):

                    plt.title('run {} of {} with params K={}, h={}, τ={}'.format(
                        self.j, int(tot), self.K, self.h, self.τ))
                    plt.plot(self.Υ[0], label=r'$\Upsilon$')
                    plt.plot(self.c[0], ls='--')
                    plt.legend()
                    plt.show()
            if self.j % int(tot/10) == 0:
                print(f'run {self.j} of {tot}: {round(self.j/tot,3)}')

            # if self.JΘ < tol:
            #     print('Tolerance reached in {} iterations. (JΘ = {})'.format(
            #         self.j, self.JΘ))
            #     break
            if self.j % 100 == 0:
                self.JθLog.append(self.JΘ)


class Adam(Network):  # Inherits from Network
    """The Adam method

    Args:
        Network (class): Super class
    """

    def __init__(self, Y0, c, K=10, h=0.05, τ=0.1, plotFreq=10000, plotProg=False):
        super().__init__(Y0, c, K, h, τ)
        self.log = {(K, h, τ): []}
        self.JθLog = []
        self.plotProg = plotProg
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
        β1 = 0.9
        β2 = 0.999
        α = 0.01
        ϵ = 1e-8
        mj = β1*mPrev + (1-β1)*gj
        vj = β2*vPrev + (1-β2)*np.square(gj)
        mHatt = mj / (1-β1**self.j)
        vHatt = vj / (1-β2**self.j)
        thetaPrev -= α*mHatt / (np.sqrt(vHatt) + ϵ)
        return thetaPrev, mj, vj

    def run(self, tol, tot, h=0.05):
        self.h = h
        while self.JΘ > tol and self.j <= tot:
            self.j += 1

            self.update()  # Update derivatives of the weights

            self.W, self.m[0], self.v[0] = self.adam(
                self.dW, self.W, self.m[0], self.v[0])
            self.b, self.m[1], self.v[1] = self.adam(
                self.db, self.b, self.m[1], self.v[1])
            self.w, self.m[2], self.v[2] = self.adam(
                self.dw, self.w, self.m[2], self.v[2])
            self.μ, self.m[3], self.v[3] = self.adam(
                self.dμ, self.μ, self.m[3], self.v[3])

            self.JΘ = 0.5 * np.linalg.norm(self.ΥmC)
            # Check progression (deleteable)

            if self.plotProg:
                if (self.j % self.plotFreq == 0) and (self.j > tot*0.8):
                    temp_x = np.linspace(0, 1, len(self.Υ[0]))
                    plt.title('run {} of {} with params K={}, h={}]'.format(
                        self.j, int(tot), self.K, self.h))
                    plt.plot(temp_x, self.Υ[0], label=r'$\Upsilon$')
                    plt.plot(temp_x, self.c, ls='--')
                    plt.legend()
                    plt.show()
            if self.j % int(tot/10) == 0:
                print(f'run {self.j} of {tot}: {round(self.j/tot,3)}')

            # if self.JΘ < tol:
            #     print('Tolerance reached in {} iterations. (JΘ = {})'.format(
            #         self.j, self.JΘ))
            #     break
            if self.j % 100 == 0:
                self.JθLog.append(self.JΘ)


class SGD():
    """Stochastic Gradient Descent
    """

    def __init__(self, method, Y0, c):
        self.method = method
        self.Y0 = Y0
        self.c = c
        self.sections = {}
        # I_tot = 4096

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
# x = np.array([np.linspace(-2, 2, 50), np.zeros((50,))])
# c = scale(Fy2(x[0]), 0, 1)
# # c = Fy2(x)
# xScaled = scale(x, 0, 1)
# maxiter = 4e4

""" Plain vanilla """
# start = dt.now()
# pv = PVGD(xScaled, c, K=15, plotFreq=int(1*1e4), plotProg=True)
# pv.run(1e-4, maxiter, h=0.185, τ=0.15)
# print(pv.JΘ)
# end = dt.now()
# print(f'elapsed time {end - start}\n\n')

# x_j = np.linspace(0, maxiter, int(maxiter/pv.logFreq))
# plt.plot(x_j, pv.JθLog)
# plt.show()

###################################### Find params #################################
# hList = np.linspace(0.05, 0.2, 21)
# τList = np.linspace(0.05, 0.15, 5)
# lastJΘ = []
# pv = PVGD(xScaled, c, K=15, plotFreq=int(1*1e4))
# r = 0
# start = dt.now()
# for h in hList:
#     for τ in τList:
#         print(f'h = {h}, τ = {τ}')
#         pv.run(1e-4, maxiter, h, τ)
#         lastJΘ.append([pv.JΘ, pv.h, pv.τ])
#         pv.restart(h=h, τ=τ)
#         r += 1
#         print(f'run {r}/{len(hList)*len(τList)} done\n')
# end = dt.now()
# print(f'\n\nelapsed time {end - start}\n\nAll J:')

# for i in lastJΘ:
#     print(i)
# minJ = lastJΘ[0].copy()

# for j in lastJΘ:
#     if minJ[0] > j[0]:
#         minJ = j.copy()

# print(f'\n\nBest J = {minJ}')

""" Adam """
# start = dt.now()
# ad = Adam(xScaled, c, K=15, plotFreq=int(1*1e4), plotProg=True)
# ad.run(1e-4, maxiter, h=0.09)
# print(ad.JΘ)
# end = dt.now()
# print(f'runtime {end - start}\n\n')
###################################### Find params #################################
# hList = np.linspace(0.05, 0.15, 20)
# lastJΘ = []
# ad = Adam(xScaled, c, K=15, plotFreq=int(1*1e4))

# r = 0
# start = dt.now()
# for h in hList:
#     print(f'h = {h}')
#     ad.run(1e-4, maxiter, h)
#     lastJΘ.append([ad.JΘ, ad.h])
#     ad.restart(h=h)
#     r += 1
#     print(f'run {r}/{len(hList)} done\n')
# end = dt.now()

# print(f'\n\nRuntime: {end - start}\n\nAll J:')

# minJ = lastJΘ[0].copy()
# for j in lastJΘ:
#     print(j)
#     if minJ[0] > j[0]:
#         minJ = j.copy()

# print(f'\n\nBest J = {minJ}')


"""Run with data batches"""
Ifrom = 250
Ito = 600
maxiter = 2e4
batch0 = gdata(0)
x = scale(batch0['P'][:, Ifrom:Ito], 0, 1)

c_v = scale(np.array([batch0['T'][Ifrom:Ito]]), 0, 1)

pv = PVGD(x, c_v)
pv.run(1e-2, maxiter)
# rerun
Ifrom = 600
Ito = 1000
x = scale(batch0['P'][:, Ifrom:Ito], 0, 1)

c_v = scale(np.array([batch0['T'][Ifrom:Ito]]), 0, 1)

pv.setY0andC(x, c_v)

pv.logic()

x_j = np.linspace(0, maxiter, int(maxiter/pv.logFreq))
plt.plot(x_j, pv.JθLog)
plt.show()
