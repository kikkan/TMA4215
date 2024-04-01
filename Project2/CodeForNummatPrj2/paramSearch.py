from classes import *


"""Functions"""


def findhAdam(m, f, t, n, tol=1e-5, maxiter=3*1e3):
    hList = np.linspace(f, t, n)
    Jθlist = []
    hErrorProof = []
    r = 0
    for h in hList:
        r += 1
        try:
            m.run(h, tol, maxiter)
            Jθlist.append(m.M.Jθ)
            hErrorProof.append(h)
            m.M.restart()
        except Exception as e:
            print(e, 'h =', h)

    plt.plot(hErrorProof, Jθlist)
    plt.xlabel('h')
    plt.ylabel(r'$J(\theta)$')
    plt.title(r'$J(\theta)$ after {} iterations with $K={}$, $I={}$'.format(m.maxiter, m.M.K, m.M.I))
    plt.show()
    bestJ = min(Jθlist)
    i = Jθlist.index(bestJ)
    print('Best h:', hErrorProof[i], ', J(θ) =', bestJ)


def findhτPVGD(m, hmM, τmM, hn, τn, tol=1e-4, maxiter=1e4):
    hList = np.linspace(hmM[0], hmM[1], hn)
    τList = np.linspace(τmM[0], τmM[1], τn)
    hEp = {}  # in case of error (error proof)
    Jθ = {}
    r = 0
    # tot = len(hList)*len(τList)

    for τ in τList:
        hEp[τ] = []
        Jθ[τ] = []
        try:
            for h in hList:
                r += 1
                m.run(h, τ, tol, maxiter)
                Jθ[τ].append(m.M.Jθ)
                hEp[τ].append(h)
                m.M.restart()
                # print(f'run {r}/{tot}')
        except Exception as e:
            print(e, 'h =', h)
    return hEp, Jθ


def plotParamSearchResults(h, Jθ, m):
    bestJ = 1e7
    for τ, h in h.items():
        plt.plot(h, Jθ[τ], label=r'$\tau={}$'.format(round(τ, 5)))
        hi = 0
        for J in Jθ[τ]:
            if J < bestJ:
                bestJ = J
                τatBest = [τ, h[hi]]
            hi += 1

    print(f'Best Jθ={bestJ} when τ={τatBest[0]}, h={τatBest[1]}')
    # plt.ylim(0, 0.1)
    plt.xlabel('h')
    plt.ylabel(r'$J(\theta)$')
    plt.title(r'$J(\theta)$ after {} iterations with $K={}$, $I={}$'.format(m.maxiter, m.M.K, m.M.I))
    plt.legend()
    plt.show()


if __name__ == "__main__":
    y100 = np.array([np.linspace(-2, 2, 100), np.linspace(-2, 2, 100)])
    y1000 = np.array([np.linspace(-2, 2, 1000), np.linspace(-2, 2, 1000)])
    y4000 = np.array([np.linspace(-2, 2, 4000), np.linspace(-2, 2, 4000)])

    c100 = F22(y100[0], y100[1])
    c1000 = F22(y1000[0], y1000[1])
    c4000 = F22(y4000[0], y4000[1])

    c100scaled = scale(c100.copy())
    c1000scaled = scale(c1000.copy())
    c4000scaled = scale(c4000.copy())

    mod10_100 = Model(y100, c100scaled, 10)
    mod15_100 = Model(y100, c100scaled, 15)
    mod20_100 = Model(y100, c100scaled, 20)

    mod10_1000 = Model(y1000, c1000scaled, 10)
    mod15_1000 = Model(y1000, c1000scaled, 15)
    mod20_1000 = Model(y1000, c1000scaled, 20)

    mod10_4000 = Model(y4000, c4000scaled, 10)
    mod15_4000 = Model(y4000, c4000scaled, 15)
    mod20_4000 = Model(y4000, c4000scaled, 20)
    """PVGD"""
    pvgd10_100 = PVGD(mod10_100.copy())
    pvgd15_100 = PVGD(mod15_100.copy())
    pvgd20_100 = PVGD(mod20_100.copy())

    pvgd10_1000 = PVGD(mod10_1000.copy())
    pvgd15_1000 = PVGD(mod15_1000.copy())
    pvgd20_1000 = PVGD(mod20_1000.copy())

    pvgd10_4000 = PVGD(mod10_4000.copy())
    pvgd15_4000 = PVGD(mod15_4000.copy())
    pvgd20_4000 = PVGD(mod20_4000.copy())

    # search
    # hminMax = (0.01, 0.4)
    # τminMax = (0.0001, 0.1)
    # h, Jθ = findhτPVGD(pvgd10_1000, hminMax, τminMax, 3, 3, tol=1e-5, maxiter=3*1e3)
    # plotParamSearchResults(h, Jθ, pvgd10_1000)

    """Adam"""
    adam10_100 = Adam(mod10_100.copy())
    adam15_100 = Adam(mod15_100.copy())
    adam20_100 = Adam(mod20_100.copy())

    adam10_1000 = Adam(mod10_1000.copy())
    adam15_1000 = Adam(mod15_1000.copy())
    adam20_1000 = Adam(mod20_1000.copy())

    adam10_4000 = Adam(mod10_4000.copy())
    adam15_4000 = Adam(mod15_4000.copy())
    adam20_4000 = Adam(mod20_4000.copy())

    # Search
    findhAdam(adam10_100, 0.05, 0.5, 4)
