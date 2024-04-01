from classes import *

import matplotlib.pyplot as plt


# We wish to find how qn, pn changes with n, and therefore how the hamiltonian changes
# along the n-values


# Examples of seperable Hamiltonian problems
# T:The kinetic energy
# V:The potensial energy


# T and V for the hamiltonian in nonlinear pendelum
def T_H_1(p):
    return (1/2)*p**2


def V_H_1(q):
    return m*g*l*(1-np.cos(q))

# T and V for the Hamiltonian in Kepler's two-body problem


def T_H_2(x):
    return (1/2)*np.dot(np.transpose(x), x)


def V_H_2(y):
    return -((1)/(np.sqrt(y[0]**2+y[1]**2)))

# T and V for the Hamiltonian in Henon-heile's problem


def T_H_3(p):
    return (1/2)*np.dot(np.transpose(p), p)


def V_H_3(q):
    return (1/2)*np.dot(np.transpose(q), q)+(q[0]**2)*q[1]-(1/3)*q[1]**3

# gradient of Hamiltonian in non linear pendelum
# The first coordinate returned is the deriate of T with respect to p and the second V with respect to q


def grad1(qn, pn):
    return pn, m*g*l*np.sin(qn)

# gradient of Hamiltonian in Kepler two-body problem for a specific q and p
# dp_1. The derivate of T. dq_1 is the derivate of V. both with two coordinates


def grad2(qn, pn):
    dT = np.zeros(2)  # dT=dT/dp a 2x1 vector
    dV = np.zeros(2)  # dV=dV/dq a 2x1 vector
    dT[0] = pn[0]  # dT[0]=p_1
    dT[1] = pn[1]  # dT[1]=p_2
    dV[0] = (qn[0])/((qn[0]**2+qn[1]**2)**(3/2))  # dV[0]
    dV[1] = ((qn[1])/((qn[0]**2+qn[1]**2)**(3/2)))
    return dT, dV

# gradient of Hamiltonian in Henon-heiles problem


def grad3(qn, pn):
    dT = np.zeros(2)  # dT=dT/dp a 2x1 vector
    dV = np.zeros(2)  # dV=dV/dp a 2x1 vector
    dT[0] = pn[0]  # dT[0]=p_1
    dT[1] = pn[1]  # dT[1]=p_2
    dV[0] = qn[0]+2*qn[0]*qn[1]  # dV[0]
    dV[1] = qn[1]+qn[0]**2-qn[1]**2  # dV[0]
    return dT, dV


def symEuler(tot_t, T, V, q0, p0, h, grad, dim, N=200):
    if dim > 1:
        A = np.zeros((2, N+1, dim))
    if dim == 1:
        A = np.zeros((2, N+1))
    A[0, 0], A[1, 0] = q0, p0

    t = np.zeros((N+1))
    H = np.zeros((N+1))
    Vlist = np.zeros((N+1))
    Tlist = np.zeros((N+1))
    Vlist[0] = V(A[0, 0])
    Tlist[0] = T(A[1, 0])
    H[0] = Tlist[0]+Vlist[0]
    for n in range(N):
        qn, pn = A[0, n], A[1, n]
        dp = grad(qn, pn)[0]
        A[0, n+1] = A[0, n]+h*dp
        qn1 = A[0, n+1]
        dq1 = grad(qn1, pn)[1]
        A[1, n+1] = A[1, n]-h*dq1

        t[n+1] = (n+1)*h
        Vlist[n+1] = V(A[0, n+1])
        Tlist[n+1] = T(A[1, n+1])
        H[n+1] = Tlist[n+1]+Vlist[n+1]
    return A, t, H, Vlist, Tlist


def symEulerOpt(h, q0, p0, gradT, gradV):
    num_p = [p0]
    num_q = [q0]
    for n in range(len(gradT)):
        num_q.append(num_q[n] + h * gradT[n])
        num_p.append(num_p[n] - h * gradV[n])
    return num_q, num_p


### Størmer verlet ###
### similar to symplectic euler except with the extra step p_(n+1/2) ###

def Stormer_verlet(tot_t, T, V, q0, p0, h, grad, dim):
    if dim == 1:
        A = np.zeros((2, N+1))
    if dim > 1:
        A = np.zeros((2, N+1, dim))  # A[0]=qn values A[1]=pn values
    A[0, 0], A[1, 0] = q0, p0
    t = np.zeros((N+1))
    H = np.zeros((N+1))
    Vlist = np.zeros((N+1))
    Tlist = np.zeros((N+1))
    Vlist[0] = V(A[0, 0])
    Tlist[0] = T(A[1, 0])
    H[0] = Tlist[0]+Vlist[0]
    for n in range(N):
        qn, pn = A[0, n], A[1, n]
        dq = grad(qn, pn)[1]
        pn_12 = pn-(h/2)*dq  # T p_(n+1/2)
        dp_12 = grad(qn, pn_12)[0]
        A[0, n+1] = qn+h*dp_12
        qn1 = A[0, n+1]
        dqn1 = grad(qn1, pn)[1]
        A[1, n+1] = pn_12-(h/2)*dqn1
        t[n+1] = (n+1)*h
        t[n+1] = (n+1)*h
        Vlist[n+1] = V(A[0, n+1])
        Tlist[n+1] = T(A[1, n+1])
        H[n+1] = Tlist[n+1]+Vlist[n+1]
    return A, t, H, Vlist


def symEulerOpt2D(N, h, q0, p0, optT, optV):
    num_p = np.zeros((N+1, optT.M.d, 1))
    num_q = np.zeros((N+1, optV.M.d, 1))
    num_q[0] = q0
    num_p[0] = p0
    for n in range(N):
        num_q[n+1] = num_q[n]+h*optT.M.computeGradYn(num_p[n].reshape(4, 1))
        num_p[n+1] = num_p[n]-h*optV.M.computeGradYn(num_q[n+1].reshape(4, 1))
    return num_q, num_p


if __name__ == "__main__":
    m = 1.0  # mass in nonlinear pendelum
    g = 9.81  # Gravitational acceleration
    l = 0.50  # length of pendelum
    N = 1000

    """Example fncs"""
    # setup
    steps = 100
    yF2 = np.linspace(-2, 2, steps)
    yFcos = np.linspace(-np.pi/3, np.pi/3, steps)
    yF22 = np.array([np.linspace(-2, 2, steps), np.linspace(-2, 2, steps)])
    yFsqrt = np.array([np.linspace(1, 2, steps), np.linspace(2, 3, steps)])

    cF2 = F2(yF2)
    cFcos = Fcos(yFcos)
    cF22 = F22(yF22[0], yF22[1])
    cFsqrt = Fsqrt(yFsqrt[0], yFsqrt[1])
    # for 1 dim we need one more dim
    yF2mod = np.array([yF2, np.zeros_like(yF2)])
    yFcosmod = np.array([yFcos, np.zeros_like(yFcos)])
    # Scale C values
    cF2scaled = scale(cF2)
    cFcosscaled = scale(cFcos)
    cF22scaled = scale(cF22)
    cFsqrtscaled = scale(cFsqrt)
    # setup models. fixing start weights
    modF2 = Model(yF2mod, cF2scaled, 10)
    modFcos = Model(yFcosmod, cFcosscaled, 10)
    modF22 = Model(yF22, cF22scaled, 10)
    modFsqrt = Model(yFsqrt, cFsqrtscaled, 10)
    # assign optimizers
    pvgdF2 = PVGD(modF2.copy())
    pvgdFcos = PVGD(modFcos.copy())
    pvgdF22 = PVGD(modF22.copy())
    pvgdFsqrt = PVGD(modFsqrt.copy())

    adamF2 = Adam(modF2.copy())
    adamFcos = Adam(modFcos.copy())
    adamF22 = Adam(modF22.copy())
    adamFsqrt = Adam(modFsqrt.copy())

    # Run
    tol = 1e-5
    maxiter = 3*1e3
    h = 0.1
    τ = 0.01

    pvgdF2.run(h, τ, tol, maxiter)
    # pvgdFcos.run(h, τ, tol, maxiter)
    # pvgdF22.run(h, τ, tol, maxiter)
    # pvgdFsqrt.run(h, τ, tol, maxiter)

    adamF2.run(h, tol, maxiter)
    # adamFcos.run(h, tol, maxiter)
    # adamF22.run(h, tol, maxiter)
    # adamFsqrt.run(h, tol, maxiter)

    plotApprox(cF2, pvgdF2, adamF2)
