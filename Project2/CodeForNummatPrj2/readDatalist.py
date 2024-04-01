import numpy as np
import os
import csv


class ReadFile:
    def __init__(self, fn):
        # Fetching data
        f = open("project_2_trajectories/datalist_batch_" + fn + ".csv", 'r')
        datalist = list(csv.reader(f))
        f.close()

        # Assigning object variables
        self.fileName = "datalist_batch_" + fn
        self.setup = datalist.pop(0)

        q1, q2, q3, p1, p2, p3 = [], [], [], [], [], []

        self.data = {"t": [], "Q": [], "P": [], "T": [], "V": []}
        for d in datalist:
            self.data['t'].append(float(d[0]))
            self.data['T'].append(float(d[7]))
            self.data['V'].append(float(d[8]))
            q1.append(float(d[1]))
            q2.append(float(d[2]))
            q3.append(float(d[3]))
            p1.append(float(d[4]))
            p2.append(float(d[5]))
            p3.append(float(d[6]))

        Q_data = np.transpose(
            np.array([[q1[i], q2[i], q3[i]] for i in range(len(datalist))]))
        P_data = np.transpose(
            np.array([[p1[i], p2[i], p3[i]] for i in range(len(datalist))]))

        self.data['Q'] = Q_data
        self.data['P'] = P_data

    def __repr__(self):
        return self.fileName

    def __len__(self):
        return len(self.data[self.setup[0]])

    def __getitem__(self, i):
        if type(i) == int and i < len(self.setup):
            return np.array(self.data[self.setup[i]])
        elif type(i) == str and i in self.data.keys():
            return np.array(self.data[i])
        else:
            raise Exception('Invalid argument for readFile object.\
                            \nValid args are {}'.format(self.data.keys()))


# fnStart = "datalist_batch_"

# batch_0 = readFile(str(0))
# # batch_1 = readFile(path + 'datallist_batch_0.csv')

# print(batch_0['t'])
# print(batch_0['q1'])
