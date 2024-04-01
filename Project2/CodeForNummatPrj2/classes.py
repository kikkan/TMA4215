import numpy as np
import datetime as dt
from miscFncs import *


class Model():
    """Base model:
        It holds startup informasjon about the model and the network used in this method.
    """

    def __init__(self, Y0, C, K, weigths=None):
        """Set values"""
        self.d = len(Y0)
        self.I = len(Y0[0])
        self.C = C  # Ska denna vær init her?
        self.K = K
        self.h = None
        self.j = 0  # Iterations
        self.Υ = np.zeros_like(self.C)
        self.Y0 = Y0
        self.ΥmC = self.Υ - self.C
        self.Jθ = 0.5 * np.linalg.norm(self.Υ - self.C)**2
        self.Jθlog = []
        if weigths:  # Imports weigths
            self.startθ = weigths
        else:  # set random weights
            self.startθ = {'W': np.random.rand(self.K, self.d, self.d),
                           'b': np.random.rand(self.K, self.d, 1),
                           'w': np.random.rand(self.d, 1),
                           'μ': np.random.rand()}
        """Memory allocation"""
        self.θ = self.copyWeights(self.startθ)
        self.Z = np.zeros((K+1, self.d, self.I))
        self.Z[0] = Y0
        # One less than intended bc P[0] is not used
        self.P = np.zeros((K, self.d, self.I))
        self.dθ = {'W': np.zeros((self.K, self.d, self.d)),
                   'b': np.zeros((self.K, self.d, 1)),
                   'w': np.zeros((self.d, 1)),
                   'μ': 0}

    def __repr__(self):
        pass

    def __len__(self):
        return self.I

    def copyWeights(self, weights):
        θ = {'W': weights['W'].copy(),
             'b': weights['b'].copy(),
             'w': weights['w'].copy(),
             'μ': weights['μ']}
        return θ

    def copy(self):
        θ = self.copyWeights(self.θ)
        return Model(self.Y0.copy(), self.C.copy(), self.K, θ)

    def computeZ(self):  # (4)
        W = self.θ['W']
        b = self.θ['b']
        h = self.h
        K = self.K
        for k in range(K):
            self.Z[k+1] = self.Z[k] + h * σ(W[k] @ self.Z[k] + b[k])

    def backwardPropagation(self):
        W = self.θ['W']
        b = self.θ['b']
        w = self.θ['w']
        μ = self.θ['μ']
        K = self.K
        Z = self.Z
        h = self.h
        K = self.K
        # Save computation
        Zxw = Z[-1].T @ w + μ
        # Update variables (Approximation)
        self.Υ = η(Zxw).T  # (5)
        self.ΥmC = self.Υ - self.C
        self.Jθ = 0.5*np.linalg.norm(self.ΥmC)**2
        # Compute last P[K]
        self.P[self.K-1] = np.outer(w, (self.ΥmC * dη(Zxw).T))  # (10) (verified)
        """Backward propagation"""
        for k in range(K-1, 0, -1):  # Change P index to K+1
            self.P[k-1] = self.P[k] + h*W[k].T @ (dσ(W[k] @ Z[k] + b[k]) * self.P[k])  # (11)

    def computeGradients(self):
        W = self.θ['W']
        b = self.θ['b']
        w = self.θ['w']
        μ = self.θ['μ']
        Z = self.Z
        P = self.P
        h = self.h
        K = self.K
        # save computation
        dηZw = dη(Z[K].T @ w + μ)
        # compute (8) and (9)
        self.dθ['μ'] = dηZw.T @ self.ΥmC.T
        self.dθ['w'] = Z[K] @ (self.ΥmC.T * dηZw)
        # Compute (12) and (13)
        for k in range(K):
            # Save computation
            PhadσWZ = h * P[k] * dσ(W[k] @ Z[k] + b[k])
            self.dθ['W'][k] = PhadσWZ @ Z[k].T
            self.dθ['b'][k] = PhadσWZ @ np.ones((self.I, 1))

    def update(self):
        self.computeZ()
        self.backwardPropagation()
        self.computeGradients()

    def setInput(self, Y0, C):
        self.Y0 = Y0.copy()
        self.d = len(Y0)
        self.I = len(Y0[0])
        self.Z = np.zeros((self.K+1, self.d, self.I))
        self.Z[0] = Y0.copy()
        self.P = np.zeros((self.K, self.d, self.I))
        self.C = C.copy()
        self.Υ = np.zeros_like(self.C)
        self.ΥmC = self.Υ - self.C
        self.Jθ = 0.5 * np.linalg.norm(self.Υ - self.C)**2
        self.j = 0

    def restart(self):
        self.θ = self.copyWeights(self.startθ)
        self.j = 0
        self.Υ = np.zeros_like(self.C)
        self.ΥmC = self.Υ - self.C
        self.Jθ = 0.5 * np.linalg.norm(self.Υ - self.C)**2

    def gradTest(self, Y0):
        Z = np.zeros_like(self.Z)
        Z[0] = Y0

    def getGrads(self, Z=np.array([])):
        h = self.h
        K = self.K
        W = self.θ['W']
        b = self.θ['b']
        w = self.θ['w']
        μ = self.θ['μ']
        if len(Z) == 0:
            Z = self.Z.copy()

        # algorithm
        A = np.dot(w, dη(np.dot(w.T, Z[K]) + μ))
        for k in range(K, 0, -1):
            A = A + W[k-1].T @ (h * dσ(W[k-1] @ Z[k-1] + b[k-1]) * A)
        return A

    def computeGradYn(self, y):
        W = self.θ['W']
        b = self.θ['b']
        K = self.K
        h = self.h
        vec = np.zeros((K+1, self.d, 1))
        vec[0] = y
        for k in range(K):
            core = h * σ(W[k] @ vec[k] + b[k])
            vec[k+1] = vec[k] + core
        return self.getGrads(vec)


class Adam():
    def __init__(self, model, h=None, tol=None, maxiter=None):
        self.M = model
        self.M.h = h
        self.tol = tol
        self.maxiter = maxiter
        self.m = {'W': np.zeros((model.K, model.d, model.d)),
                  'b': np.zeros((model.K, model.d, 1)),
                  'w': np.zeros((model.d, 1)),
                  'μ': 0}
        self.v = self.m.copy()

    def __repr__(self):
        # re = f'Adam, h={self.M.h}, K={self.M.K}, iter={self.maxiter}, Jθ={round(self.M.Jθ, 5)}'
        re = 'Adam'
        return re

    def name(self):
        return 'Adam'

    def algo(self):
        β1 = 0.9
        β2 = 0.999
        α = 0.01
        ϵ = 1e-8
        for key in self.m.keys():
            self.m[key] = β1*self.m[key] + (1-β1)*self.M.dθ[key]
            self.v[key] = β2*self.v[key] + (1-β2)*np.square(self.M.dθ[key])

            m = self.m[key] / (1 - β1**self.M.j)
            v = self.v[key] / (1 - β2**self.M.j)
            self.M.θ[key] -= α * m / (np.sqrt(v) + ϵ)

    def run(self, h, tol, maxiter):
        self.M.h = h
        self.tol = tol
        self.maxiter = maxiter
        self.M.update()
        # while self.M.Jθ > tol and self.M.j <= maxiter:
        while self.M.j <= maxiter:
            self.M.j += 1
            self.algo()
            self.M.update()
        self.M.Jθlog.append([self.M.h, self.M.Jθ])

    def setParams(self, h, tol, maxiter):
        self.h = h
        self.tol = tol
        self.maxiter = maxiter

    def _restart_m_and_v(self):
        self.m = {'W': np.zeros((self.M.K, self.M.d, self.M.d)),
                  'b': np.zeros((self.M.K, self.M.d, 1)),
                  'w': np.zeros((self.M.d, 1)),
                  'μ': 0}
        self.v = self.m.copy()

    def continueRun(self):
        # self._restart_m_and_v()
        self.M.j = 0
        self.run(self.M.h, self.tol, self.maxiter)


class PVGD():
    """Plain Vanilla Gradient Descent optimizer
    """

    def __init__(self, model):
        self.M = model
        self.τ = None
        self.maxiter = None
        self.log = {}

    def __repr__(self):
        # re = f'PVGD, h={self.M.h}, τ={self.τ}, K={self.M.K}, iter={self.M.j}, Jθ={round(self.M.Jθ, 5)}'
        re = 'PVGD'
        return re

    def name(self):
        return 'PVGD'

    def run(self, h, τ, tol, maxiter):
        self.M.h = h
        self.τ = τ
        self.maxiter = maxiter
        # self.log[(τ, maxiter)] = []
        while self.M.Jθ > tol and self.M.j < maxiter:
            self.M.j += 1
            self.M.update()
            for key in self.M.θ:
                self.M.θ[key] -= self.τ * self.M.dθ[key]
                # if self.M.j % 10 == 0:
                #     self.log[(τ, maxiter)].append(self.M.Jθ)
            # self.plotProg()

    def plotProg(self):
        if self.M.j % 1000 == 0:
            # plt.plot(self.log[(self.τ, self.maxiter)][:][1], self.log[(self.τ, self.maxiter)][:][0])
            plt.plot(self.M.γ)
            plt.title(f'{self.log.keys()}')
            plt.show()


class SGD():
    """Stochastic Gradient Descent method on chosen optimizer
    """

    def __init__(self, opt, n=None):
        # Extract the total input and exact output
        self.opt = opt
        self.M = opt.M
        self.Yt = opt.M.Y0
        self.Ct = opt.M.C
        self.maxiter = opt.maxiter
        # SGD's own variables
        self.jt = 0
        self.subsets = {}
        self.n = n

    def __repr__(self):
        pass

    def separate(self, n):
        self.n = n
        self.opt.maxiter = int(self.maxiter/n)
        for i in range(n):
            self.subsets[i] = []

        YtCt = []
        for i in range(len(self.Ct)):
            YtCt.append([self.Yt[:, i], self.Ct[i]])

        i = 0
        while YtCt:
            r = np.random.randint(0, len(YtCt))
            self.subsets[i].append(YtCt.pop(r))
            i += 1
            if i == self.n:
                i = 0

    def run(self):

        for YC in self.subsets.values():
            Y0 = []
            C = []
            for YCi in YC:
                Y0.append(YCi[0])
                C.append(YCi[1])
            Y0 = np.array(Y0).T
            C = np.array(C)
            self.M.setInput(Y0, C)
            self.opt.continueRun()
        self.M.setInput(self.Yt, self.Ct)
        self.M.update()

    def continueRun(self):
        self.separate(self.n)
        self.run()

    def name(self):
        return 'SGD ' + self.opt.name()

    def setInput(self, Y0, C):
        self.Yt = Y0.copy()
        self.Ct = C.copy()
