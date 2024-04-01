import numpy as np
import matplotlib.pyplot as plt
import random as rnd
from datetime import datetime as dt

from givenFncs import generate_data

# Deleteable. Used for debug
np.seterr(all='raise')

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


"""##################### Classes"""


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

    def computeZ(self, W, b, h, K):  # (4)
        for k in range(K):
            self.Z[k+1] = self.Z[k] + h * σ(W[k] @ self.Z[k] + b[k])

    def backwardPropagation(self, W, b, w, μ, Z, h, K):
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

    def computeGradients(self, W, b, w, μ, Z, P, h, K):
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
        self.computeZ(self.θ['W'], self.θ['b'], self.h, self.K)
        self.backwardPropagation(self.θ['W'], self.θ['b'], self.θ['w'],
                                 self.θ['μ'], self.Z, self.h, self.K)
        self.computeGradients(self.θ['W'], self.θ['b'], self.θ['w'],
                              self.θ['μ'], self.Z, self.P, self.h, self.K)

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

    def getGrads(self, Z):
        h = self.h
        K = self.K
        W = self.θ['W']
        b = self.θ['b']
        w = self.θ['w']
        μ = self.θ['μ']
        Z = self.Z.copy()

        # algorithm
        A = np.dot(w, dη(np.dot(w.T, Z[K]) + μ))
        for k in range(K, 0, -1):
            A = A + W[k-1].T @ (h * dσ(W[k-1] @ Z[k-1] + b[k-1]) * A)
        return A


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
        self._restart_m_and_v()
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

    def run(self, h, τ, tol, maxiter, ):
        self.M.h = h
        self.τ = τ
        self.maxiter = maxiter
        self.log[(τ, maxiter)] = []
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
        self.opt.maxiter = int(self.maxiter / n)
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

    def continueRun(self):
        self.separate(self.n)
        self.opt.continueRun()

    def name(self):
        return 'SGD ' + self.opt.name()


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
            lbl += ', h={}, τ={}, J(θ)={}'.format(mod.M.h, mod.τ, round(mod.M.Jθ, 3))
        else:
            lbl += ', h={}, J(θ)={}'.format(mod.M.h, round(mod.M.Jθ, 3))

        plt.plot(x, scaleBack(mod.M.Υ[0], C), label=lbl)

    if C.shape[0] == 1:
        plt.plot(x, C[0], label='F(y)', ls='--')
    else:
        plt.plot(x, C, label='F(y)', ls='--')
    plt.title(r'Comparing $\tilde{F}(y)$ with $F(y)$')
    plt.legend()
    plt.show()


def train(opt, bfrom, bTo, dFrom, dTo, inp='Q', output='V'):
    print(f'Training on batch {bfrom}-{bTo} with {dTo - dFrom} datapoints.\n'
          f'Inputs: {inp}, output: {output}, iterations: {opt.maxiter}, Layers: {opt.M.K}')
    print('Batch\tJ(θ)')
    start = dt.now()
    for i in range(bfrom, bTo+1):
        try:
            batch = generate_data(i)
            Y0 = np.array(batch[inp][:, dFrom:dTo])
            C = np.array(batch[output][dFrom:dTo])
            Cscaled = scale(C.copy())
            opt.M.setInput(Y0, Cscaled)
            opt.continueRun()
            print(f'{i}\t{opt.M.Jθ}')
            # t = batch['t'][dFrom:dTo] # used for plot approx
            # plotApprox(C, opt, x=t)
        except Exception as exc:
            print('Error occured on batch {}:\n{}'.format(i, exc))
    end = dt.now()
    print('Training time:', end-start)


def testOnNew(Y, C, t, m):
    Z = Y.copy()
    θ = m.M.θ

    for k in range(m.M.K):
        Z = (Z + m.M.h * σ(θ['W'][k] @ Z + θ['b'][k]))

    Υ = η(Z.T @ θ['w'] + θ['μ']).T[0]

    Jθ = 0.5*np.linalg.norm(Υ - scale(C))**2

    plt.title(r'Trained on new input, $J(\theta) = {}$'.format(round(Jθ, 4)))
    plt.plot(t, scaleBack(Υ, C), label=r'$\tilde{F}(y)$')
    plt.plot(t, C, ls='--')
    plt.legend()
    plt.show()


def plotLogic(opt):
    pass


if __name__ == '__main__':
    np.random.seed()
    """Run with ex fncs"""
    # # Setup
    # I = 50
    # maxiter = 1e4
    # x = np.array([np.linspace(-2, 2, I), np.zeros((I,))])
    # C = F2(x[0])
    # Cscaled = scale(C.copy(), 0, 1)

    # # run
    # mod1 = Model(x, Cscaled, 15)
    # # PVGD
    # pvgdOpt = PVGD(mod1.copy())
    # pvgdOpt.run(0.05, 0.15, 1e-4, maxiter)
    # # Adam
    # adamOpt = Adam(mod1.copy())
    # adamOpt.run(0.05, 1e-4, maxiter)
    # # show
    # plotApprox(C, pvgdOpt, adamOpt, x=x[0])

    """Run with batch"""
    Ifrom = 0
    Ito = 100
    maxiter = 1e4
    tol = 1e-9

    # Training 1
    batch0 = generate_data(0)

    # y0 = scale(batch0['P'][:, Ifrom:Ito])
    y0 = batch0['P'][:, Ifrom:Ito]
    t0 = scale(batch0['t'][Ifrom:Ito], 0, 1)
    C_T1 = np.array([batch0['T'][Ifrom:Ito]])
    C_T1scaled = scale(C_T1, 0, 1)
    # Setup model
    modT = Model(y0, C_T1scaled, 10)
    # PVGD
    PVGDOptBatch = PVGD(modT.copy())

    """Fin best params for h"""
    # findhτPVGD(PVGDOptBatch, (0.01, 0.5), (0.001, 0.2), 3, 3)
    # PVGDOptBatch.run(0.1, 0.01, tol, maxiter)
    # # adam
    # adamOptBatch = Adam(modT.copy())
    # adamOptBatch.run(0.5, tol, maxiter)

    # plotApprox(C_T1, PVGDOptBatch, x=t0)
    # plotApprox(C_T1, adamOptBatch, x=t0)
    # plotApprox(C_T1, PVGDOptBatch, adamOptBatch, x=t0)

    # Training 2
    # batch1 = generate_data(1)
    # y1 = scale(batch1['P'][:, Ifrom:Ito], 0, 1)
    # t1 = scale(batch1['t'][Ifrom:Ito], 0, 1)
    # C_T2 = np.array([batch1['T'][Ifrom:Ito]])
    # C_T2scaled = scale(C_T2, 0, 1)
    # adamOptBatch.M.setInput(y1, C_T2scaled)
    # adamOptBatch.run(0.5, tol, maxiter)
    # plotApprox(C_T2, adamOptBatch, x=t1)

    # # Test on new dataset
    # batchTest = generate_data(2)
    # yTest = scale(batchTest['P'][:, Ifrom:Ito], 0, 1)
    # tTest = scale(batchTest['t'][Ifrom:Ito], 0, 1)
    # C_Ttest = np.array([batchTest['T'][Ifrom:Ito]])
    # C_TtestScaled = scale(C_Ttest, 0, 1)
    # testOnNew(yTest, C_Ttest, tTest, adamOptBatch)

    """Find params"""
    # Find best h
    n = 100
    r = 0
    # start = dt.now()
    # findhAdam(adamOpt, 0.01, 0.5, n)
    # end = dt.now()
    # print(f'runtime: {end - start}')
