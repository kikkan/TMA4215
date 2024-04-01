from allNew import *


def TH1(p):
    """pendulum kinetic

    Args:
        p (float): momentum

    Returns:
        float: kinetic energy
    """
    return (1/2)*p**2


def VH1(q):
    """pendulum potensial

    Args:
        q (float): position

    Returns:
        float: potensial energy
    """
    return m*g*l*(1-np.cos(q))


def gradVT1(qn, pn):  # Tar denna virkelig inn motsatt av åssn den gir d ut? G(q,p) = (T(p), V(q))
    return pn, m*g*l*np.sin(qn)


def TH2(x):
    return (1/2)*np.dot(np.transpose(x), x)


def VH2(y):
    return -((1)/(np.sqrt(y[0]**2+y[1]**2)))


def grad2old(qn, pn):
    A_1 = np.zeros(2)
    B_1 = np.zeros(2)
    print(pn)
    A_1[0] = pn[0]
    A_1[1] = pn[1]
    B_1[0] = (qn[0])/((qn[0]**2+qn[1]**2)**(3/2))
    B_1[1] = ((qn[1])/((qn[0]**2+qn[1]**2)**(3/2)))
    print(A_1)
    return A_1, B_1


def grad2(q1, q2, p1, p2):
    dQ = np.zeros((2, len(q1)))
    dP = np.zeros((2, len(q1)))
    dP[0] = q1
    dP[1] = q2

    dQ[0] = (q1)/((q1**2+q2**2)**(3/2))
    dQ[1] = ((q2)/((q1**2+q2**2)**(3/2)))
    return dP, dQ


def findGrad(Opt):
    # extracting paramters
    h = Opt.M.h
    K = Opt.M.K

    Z = Opt.M.Z.copy()
    W = Opt.M.θ['W']
    b = Opt.M.θ['b']
    w = Opt.M.θ['w']
    μ = Opt.M.θ['μ']

    # algorithm
    A = np.dot(w, dη(np.dot(w.T, Z[K]) + μ))
    for k in range(K, 0, -1):
        A = A + W[k-1].T @ (h * dσ(W[k-1] @ Z[k-1] + b[k-1]) * A)
    return A


"""gradient nonlinear pendulum (1D)"""
# Setup
np.random.seed()
m = 1
g = 9.81
l = 2
I = 100
maxiter = 1e4
x = np.array([np.linspace(0, 10, I), np.zeros((I,))])
C_T = TH1(x[0])
p = np.linspace(1, 10, I)
q = np.linspace(1, 10, I)
dV, dT = gradVT1(q, p)  # ???
# Kinetic
cP = TH1(x[0])
modT = Model(x, scale(C_T), 20)  # create instance of model
adamOptT = Adam(modT.copy())
adamOptT.run(0.05, 1e-4, maxiter)
gradientT = adamOptT.M.getGrads()
# Plot
plt.figure()
plt.plot(x[0], dV, label="T exact", ls='--')
plt.plot(x[0], scaleBack(gradientT[0], C_T), label='Approx')
plt.legend()

# Potential
C_V = VH1(x[0])
modV = Model(x, scale(C_V), 20)  # create instance of model
adamOptV = Adam(modV.copy())
adamOptV.run(0.05, 1e-4, maxiter)
gradientV = adamOptV.M.getGrads()  # (2xI)
gradientV = findGrad(adamOptV)  # (2xI)
# plot
plt.figure()
plt.plot(x[0], C_V, label="V exact")
plt.plot(x[0], scaleBack(gradientV[0], C_V))

plt.legend()
plt.show()
