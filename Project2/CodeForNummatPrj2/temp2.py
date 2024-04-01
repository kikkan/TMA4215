import numpy as np
import random as rnd
import matplotlib.pyplot as plt
from datetime import datetime as dt

# a = np.random.rand(1, 20)

# sec = {}

# parts = 4
# for i in range(parts):
#     sec[i] = [[], []]

# b = [i for i in range(len(a[0]))]

# # for i in range(len(a[0])):
# #     r = np.random.randint(0, len(b))
# #     sec[i % (parts)][0].append(b.pop(r))
# #     sec[i % (parts)][1].append(a[0][sec[i % (parts)][0][-1]])
# i = 0
# while b:
#     r = np.random.randint(0, len(b))
#     sec[i % (parts)][0].append(b.pop(r))
#     sec[i % (parts)][1].append(a[0][sec[i % (parts)][0][-1]])
#     i += 1

# print('done')

# for key, el in sec.items():
#     print('####### Key {} ########'.format(key))
#     for i, v in zip(el[0], el[1]):
#         print('{}\t{}\t{}'.format(i, v, a[0][i]))


# a = np.random.rand(4, 4)
# print(a, '\n\n')

# for i in range(4):
#     for j in range(4):
#         if np.random.rand() > .5:
#             a[i, j] *= -1

# print((a*a))

# ax = plt.gca()
# color = next(ax._get_lines.prop_cycler)['color']
# i = 0
# for i in range(10):
#     y = []
#     x = []
#     if i % 2 == 1:
#         color = next(ax._get_lines.prop_cycler)['color']
#     for j in range(10):
#         y.append(j**2)
#         i += 1
#         x.append(i)
#     plt.plot(x, y, color=color)
#     # plt.plot(x, y)
# plt.show()

# color=next(ax._get_lines.prop_cycler)['color']

def timeIt():
    i = 0
    while True:
        if i == 0:
            s = dt.now()
            i = 1
            yield None
        else:
            print(dt.now() - s)
            i = 0
            yield None


a = timeIt()
for i in range(10000):
    for j in range(10):
        a = 0
a = timeIt()
