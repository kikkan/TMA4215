import os
import csv
from datetime import datetime as dt  # To check runtime for educational purposes
import numpy as np
import matplotlib.pyplot as plt
from readDatalist import ReadFile
from mpl_toolkits.mplot3d import Axes3D


# Example functions
def generate_data(batch=0):
    """Generates one dict with data from one batch

    Args:
        batch (int, optional): Batch number. Defaults to 0.

    Returns:
        Dictionary: Keys: t, Q, P, T, V
    """

    start_path = ""
    path = start_path+"project_2_trajectories/datalist_batch_" + \
        str(batch)+".csv"
    with open(path, newline="\n") as file:
        reader = csv.reader(file)
        datalist = list(reader)

    N = len(datalist)
    t_data = np.array([float(datalist[i][0]) for i in range(1, N)])
    Q1_data = [float(datalist[i][1]) for i in range(1, N)]
    Q2_data = [float(datalist[i][2]) for i in range(1, N)]
    Q3_data = [float(datalist[i][3]) for i in range(1, N)]
    P1_data = [float(datalist[i][4]) for i in range(1, N)]
    P2_data = [float(datalist[i][5]) for i in range(1, N)]
    P3_data = [float(datalist[i][6]) for i in range(1, N)]
    T_data = np.array([float(datalist[i][7]) for i in range(1, N)])
    V_data = np.array([float(datalist[i][8]) for i in range(1, N)])

    Q_data = np.transpose(
        np.array([[Q1_data[i], Q2_data[i], Q3_data[i]] for i in range(N-1)]))
    P_data = np.transpose(
        np.array([[P1_data[i], P2_data[i], P3_data[i]] for i in range(N-1)]))

    return {"t": t_data, "Q": Q_data, "P": P_data, "T": T_data, "V": V_data}


fnStart = "datalist_batch_"
s = dt.now()
batch_0 = ReadFile(str(6))
e1 = dt.now()

batch_01 = generate_data()
e2 = dt.now()
print('mine: {}\n'
      'given: {}'.format(e1-s, e2-e1))

# print(len(batch_0))
# print(batch_0['t'])
# print(batch_0['T'])
# print(batch_0['V'])


# plt.plot(batch_0['t'], batch_0['T'], label='Kinetic')
# plt.plot(batch_0['t'], batch_0['V'], label='Potential')
# plt.legend()
# plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(batch_0['Q'][0], batch_0['Q'][1], batch_0['Q'][2])
ax.plot(batch_0['Q'][0, 0], batch_0['Q'][1, 0], batch_0['Q'][2, 0], 'or', label=r'$Q_0$')
plt.legend()
plt.show()
