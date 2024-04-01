from classes import *
from supplementedFncs import generate_data


def train(opt, bfrom, bTo, dFrom, dTo, inp='Q', output='V'):
    print(f'Training on batch {bfrom}-{bTo} with {dTo - dFrom} datapoints.\n'
          f'Input: {inp}, output: {output}, iterations: {opt.maxiter}, Layers: {opt.M.K}')
#     print('Batch\tJ(θ)')
    Jθlist = []
    start = dt.now()
    for i in range(bfrom, bTo+1):
        # try:
        batch = generate_data(i)
        Y0 = np.array(batch[inp][:, dFrom:dTo])
        C = np.array(batch[output][dFrom:dTo])
        Cscaled = scale(C.copy())
        if opt.name().startswith('SGD'):
            opt.setInput(Y0, C)
        else:
            opt.M.setInput(Y0, Cscaled)
        opt.continueRun()
        Jθlist.append(opt.M.Jθ)
#             print(f'{i}\t{round(opt.M.Jθ, 5)}')
        # t = batch['t'][dFrom:dTo] # used for plot approx
        # plotApprox(C, opt, x=t)
        # except Exception as exc:
        #     print('Error occured on batch {}:\n{}'.format(i, exc))
    end = dt.now()
    print('Training time:', end-start)
    return Jθlist

# batch = generate_data(0)
# Y0Q = batch['Q']
# CV = batch['V']

# mod = Model(Y0Q, scale(CV), 30)
# adamOpt = Adam(mod.copy(), h=0.3, tol=1e-5, maxiter=1e4)
# sgdAdamOpt = SGD(adamOpt, 10)
# sgdAdamOpt.separate(10)
# sgdAdamOpt.run()

# plotApprox(CV, sgdAdamOpt)


modV = Model(np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]), np.array([1, 2, 3]), 30)
adamOptV = Adam(modV.copy(), h=0.3, tol=1e-5, maxiter=300)
sgdAdamOptV = SGD(adamOptV, 10)

batchFrom = 0  # included
batchTo = 0  # included
dataPointsFrom = 0
dataPointsTo = 4096
"""Potential"""
JθV = train(sgdAdamOptV, batchFrom, batchTo, dataPointsFrom, dataPointsTo)
