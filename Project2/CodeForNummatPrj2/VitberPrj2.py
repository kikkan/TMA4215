import matplotlib.pyplot as plt
import random as rnd
import numpy as np


# Todo Oppgitte funksjoner
def sigma(x: float):
    return np.tanh(x)


def dSigmadx(x):
    return 1 / np.cosh(x) ** 2


def eta(x):
    return (1 + np.tanh(x / 2)) / 2


def detadx(x):
    return 1/(2 * np.cosh(x) + 2)


def Z(YK, omega, mu):
    return eta(YK.T @ omega + mu*en)


def J(c, YK, omega, mu):
    return np.sum(abs(Z(YK, omega, mu) - c) ** 2) / 2


def PK(omega, Zmc, YK, mu):
    return omega @ ((Zmc * detadx(YK.T @ omega + mu)).T)


def PkPrev(P_kList, h, W_k, Y_kList, b_kList):
    tempPk = P_kList[-1]
    for k in range(1, K + 1):
        easeOfRead = dSigmadx(W_k[-k] @ Y_kList[-k] + b_kList[-k])
        tempPk = tempPk + h * W_k[-k].T @ (easeOfRead * tempPk)
        P_kList[-k - 1] = tempPk
    return P_kList


# todo Divergens
def dJdmu(YK, omega, mu, zminusc):
    return detadx(YK.T @ omega + mu * en).T @ zminusc


def dJdomega(YK, omega, mu, zminusc):
    return YK @ (zminusc * detadx(YK.T @ omega + mu))


def dJdW(h, P_kList, W, Y_kList, b_k, djdW):
    for k in range(0, K):
        djdW[k] = (h * (P_kList[k + 1] * dSigmadx(W[k] @
                                                  Y_kList[k] + b_k[k])) @ Y_kList[k].T)
    return np.array(djdW)


def dJdb(h, P, W, Y, b, djdb):
    for k in range(K):
        djdb[k] = h * (P[k + 1] * dSigmadx(W[k] @ Y[k] + b[k])) @ en
    return djdb


def algoritme(W, b, omega, mu, Y, P, h, K, djdW, djdb, j, m=[], v=[], adam=False, labels=[]):
    for k in range(0, K):
        Y[k + 1] = Y[k] + h * sigma(W[k] @ Y[k] + b[k])

    z = Z(Y[-1], omega, mu)
    Zmc = z - labels
    J = 0.5 * np.linalg.norm(Zmc) ** 2

    if bilder:
        if j % 10 == 0 or j == 1:
            print('J_{} = {}'.format(int(j), J))
    else:
        if j % 10000 == 0 or j == 1:
            print('J_{}k = {}'.format(int(j/1000), J))

    # Forrige PK
    P[-1] = PK(omega, Zmc, Y[-1], mu)
    P = PkPrev(P, h, W, Y, b)  # (K+1, 2, I)

    # Divergenser
    # Tomme arrays
    djdmu = dJdmu(Y[-1], omega, mu, Zmc)
    djdomega = dJdomega(Y[-1], omega, mu, Zmc)
    # Finner divergens
    djdW = dJdW(h, P, W, Y, b, djdW)
    djdb = dJdb(h, P, W, Y, b, djdb)

    if adam:
        newW, m[0], v[0] = Adam(djdW, W, m[0], v[0], j)
        newb, m[1], v[1] = Adam(djdb, b, m[1], v[1], j)
        newomega, m[2], v[2] = Adam(djdomega, omega, m[2], v[2], j)
        newmu, m[3], v[3] = Adam(djdmu, mu, m[3], v[3], j)
        return newW, newb, newomega, newmu[0][0], m, v, J
    else:
        newW = W - tau * djdW
        newb = b - tau * djdb
        newomega = (omega - tau * djdomega)
        newmu = (mu - tau * djdmu)
        return newW, newb, newomega, newmu, J


def generations(W, b, omega, mu, h, K, J=5000, adam=False, bilder=False, plot=False):
    if bilder:
        print('Kj�rer med bilder')
        Y0, labels = get_dataset(dataset="training",
                                 path=r"C:\PycharmProjects\Mekfys\4.Semester\Vitber\Prosjekt_2_Deep_Learning")
        labels = labels[0:I]
        Y0 = Y0[:, 0:I]
    else:
        Y0, labels = get_data_spiral_2d(I)
    labels = np.array(labels)

    # Setter av minne til de store listene
    P = np.empty((K + 1, d, I))
    Y = np.empty((K + 1, d, I))
    djdW = np.empty((K, d, d))
    djdb = np.empty((K, d, 1))

    Y[0] = Y0

    Wnew, bNew, omegaNew, muNew = W, b, omega, mu

    Jl = []
    genl = []

    if adam:
        j = 1
        m = [np.zeros((K, d, d)), np.zeros((K, d, 1)), np.zeros((d, 1)), m0]
        v = [np.zeros((K, d, d)), np.zeros((K, d, 1)), np.zeros((d, 1)), v0]
        while j < J + 1:
            Wnew, bNew, omegaNew, muNew, m, v, Jr = algoritme(Wnew, bNew, omegaNew, muNew, Y, P, h, K,
                                                              djdW, djdb, j, m, v, adam, labels)
            Jl.append(Jr)
            genl.append(j)
            j += 1

    else:
        j = 0
        while j < J:
            Wnew, bNew, omegaNew, muNew = algoritme(Wnew, bNew, omegaNew, muNew,
                                                    Y, P, h, K, djdW, djdb, j)
            Jl.append(Jr)
            genl.append(j)
            j += 1

    for k in range(0, K):
        Y[k + 1] = Y[k] + h * sigma(Wnew[k] @ Y[k] + bNew[k])

    z = Z(Y[-1], omegaNew, muNew)
    Zmc = z - labels

    feil = 0
    for i in Zmc:
        if 0.5 <= abs(i):
            feil += 1

    print('\nfeil: {}%\n'.format(feil / I * 100))

    if not bilder and plot:
        plot_progression(Y, labels)
    return Wnew, bNew, omegaNew, muNew, Jl, genl


# ADAM descent algorithm for en parameter
def Adam(gj, UPrev, mPrev, vPrev, j):
    mj = b1*mPrev + (1-b1)*gj
    vj = b2*vPrev + (1-b2)*(gj*gj)
    mHatt = mj / (1-b1**j)
    vHatt = vj / (1-b2**j)
    Uj = UPrev - a*mHatt / (np.sqrt(vHatt) + epsilon)
    return Uj, mj, vj
