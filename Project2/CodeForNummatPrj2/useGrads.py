from classes import *
from numerical import *

""" gradient kepler two-body 2D"""
tot_t = 10
h = 0.2
tol = 1e-4
maxiter = 3*1e3
N = 200
dt = tot_t/N

I2 = 200
p1 = np.array(np.linspace(1, 3, I2))
p2 = np.array(np.linspace(1, 3, I2))
q1 = np.array(np.linspace(1, 3, I2))
q2 = np.array(np.linspace(1, 3, I2))


p2Mod = np.array([p1, p2, np.zeros_like(p1), np.zeros_like(p2)])
q2Mod = np.array([q1, q2, np.zeros_like(q1), np.zeros_like(q2)])

T = np.zeros((I2))
V = np.zeros((I2))

for i in range(I2):
    T[i] = T_H_2(np.array([p1[i], p2[i]]))
    V[i] = V_H_2(np.array([q1[i], q2[i]]))

mod2T = Model(p2Mod, scale(T), 20)
adam2T = Adam(mod2T)
mod2V = Model(q2Mod, scale(V), 20)
adam2V = Adam(mod2V)
adam2T.run(h, tol, maxiter)
print(adam2T.M.Jθ)
adam2V.run(h, tol, maxiter)
print(adam2V.M.Jθ)

plotApprox(T, adam2T)


q02 = np.array([1-0.5, 0, 0, 0]).reshape(4, 1)
p02 = np.array([0, np.sqrt((1+0.5)/(1-0.5)), 0, 0]).reshape(4, 1)


num2Q, num2P = symEulerOpt2D(N, dt, q02, p02, adam2T, adam2V)

q02sym = np.array([1-0.5, 0])
p02sym = np.array([0, np.sqrt(((1+0.5)/(1-0.5)))])

C, t_3, H_2, V_2, T_2 = symEuler(tot_t, T_H_2, V_H_2, q02sym, p02sym, h, grad2, 2, N)
# q = C[0] , p = C[1]


plt.figure()
plt.plot(num2Q[:, 0], num2P[:, 0])
plt.plot(C[0][:, 0], C[1][:, 0])
plt.show()
print('Done')
