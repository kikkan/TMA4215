from allNew import *

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


def symEuler(tot_t, T, V, q0, p0, h, grad1, grad2, grad, dim, trained):
    # t, T, V, ... , gradV=None, gradT=None, gradFunc=None
    # dimension
    # A stores the values of q and p. qn=A[0], pn=A[1]
    if dim > 1:
        A = np.zeros((2, N+1, dim))
    if dim == 1:
        A = np.zeros((2, N+1))
    A[0, 0], A[1, 0] = q0, p0  # Set the start values.
    # Memory allocation
    t = np.zeros((N+1))  # list of the time values
    H = np.zeros((N+1))  # list of the Hamiltonians
    Vlist = np.zeros((N+1))  # list of potensial energy
    Tlist = np.zeros((N+1))  # list if kinteic energy
    Vlist[0] = V(A[0, 0])  # set the start potensial energy
    Tlist[0] = T(A[1, 0])  # Set the start kinteic energy
    H[0] = Tlist[0]+Vlist[0]  # Set the first hamiltonian H[0]
    for n in range(N):
        qn, pn = A[0, n], A[1, n]  # Set the values of qn, and pn
        if trained == 0:
            dp = grad(qn, pn)[0]  # dT/dp
        else:
            dp = grad1[1, n]
        A[0, n+1] = A[0, n]+h*dp  # Set q_n+1
        qn1 = A[0, n+1]  # qn_n+1
        if trained == 0:
            dq1 = grad(qn1, pn)[1]
        else:
            dq1 = grad2[0, n+1]
        A[1, n+1] = A[1, n]-h*dq1  # Set p_n+1

        t[n+1] = (n+1)*h  # the next time value
        Vlist[n+1] = V(A[0, n+1])  # Next value of V
        Tlist[n+1] = T(A[1, n+1])  # Next value of T
        H[n+1] = Tlist[n+1]+Vlist[n+1]  # Calculate the next H=T+V
    return A, t, H, Vlist


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


if __name__ == "__main__":
    """ Some setup"""
    # setup
    N = 500  # number of n
    tot_t = 10  # total time
    h = tot_t/N  # timestep
    m = 1.0  # mass in nonlinear pendelum
    g = 9.81  # Gravitational acceleration
    l = 0.50  # length of pendelum

    q0_n = 1  # starting value for q in nonlinear pendelum
    p0_n = 0  # Starting value for p in nonlinear pendelum
    q0_k, p0_k = np.array([1-0.5, 0]), np.array([0, np.sqrt(((1+0.5)/(1-0.5)))])  # Starting values for q and p in two kepler body
    # The calculated starting value p10
    p10 = np.sqrt(2*(1/12)-2*(7/1500)-0.1**2)
    q0_h, p0_h = np.array([0, 0.1]), np.array([p10, 0.1])

    """Some run"""
    A, t_1, H_0, V_0 = symEuler(tot_t, T_H_1, V_H_1, q0_n, p0_n, h, 0, 0, grad1, 1, 0)
    B, t_2, H_1, V_1 = Stormer_verlet(tot_t, T_H_1, V_H_1, q0_n, p0_n, h, grad1, 1)

    """Some plot"""
    plt.figure()
    plt.grid()
    plt.title("p plotted against q in nonlinear pendelum")
    plt.xlabel("q")
    plt.ylabel("p")
    plt.plot(A[0, :], A[1, :], label="Symplectic Euler")
    plt.plot(B[0, :], B[1, :], label="Størmer verlet method")
    plt.legend()
    plt.show()

    q = A[0]
    p = A[1]

    qMod = np.array([q, np.zeros_like(q)])
    pMod = np.array([q, np.zeros_like(q)])

    T = T_H_1(p.copy())
    V = V_H_1(q.copy())

    modT = Model(pMod, T, 10)
    adamT = Adam(modT)
    modV = Model(qMod, V, 10)
    adamV = Adam(modV)

    tol = 1e-4
    maxiter = 1e4
    h = 0.5
    dt = tot_t / N

    adamT.run(h, tol, maxiter)
    adamV.run(h, tol, maxiter)

    gradT = adamT.M.getGrads()
    gradV = adamV.M.getGrads()

    # gradT = scaleBack(gradT[0], T)
    # gradV = scaleBack(gradV[0], V)
    # print(gradT.shape)

    numQ, numP = symEulerOpt(dt, q[0], p[0], gradT[0], gradV[0])

    # numQ = np.array(numQ)
    # numP = np.array(numP)
    """With scaleBack"""
    # numQ = scaleBack(np.array(numQ), q, -1, 1)
    # numP = scaleBack(np.array(numP), p, -2, 2)
    # numQ = scaleBack(np.array(numQ), q, -2, 2)
    # numP = scaleBack(np.array(numP), p, -1, 1)
    """With Scale"""
    numQscaled = scale(np.array(numQ), min(q), max(q))
    numPscaled = scale(np.array(numP), min(p), max(p))

    # plt.plot(numQ[100:], numP[:-100])
    # plt.plot(numP[:-100], numQ[100:])
    plt.plot(numQscaled, numPscaled)
    plt.plot(q, p, label='Excat', ls='--')

    plt.legend()
    plt.show()
