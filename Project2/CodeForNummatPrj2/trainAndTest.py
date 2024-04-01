from allNew import *

"""This fnc is written in main"""
# def train(opt, bfrom, bTo, dFrom, dTo, inp='Q', output='V'):
#     for i in range(bfrom, bTo+1):
#         batch = generate_data(i)
#         Y0 = np.array(batch[inp][:, dFrom:dTo])
#         C = np.array(batch[output][dFrom:dTo])
#         Cscaled = scale(C.copy())
#         t = batch['t'][dFrom:dTo]
#         opt.M.setInput(Y0, Cscaled)
#         opt.continueRun()
#         plotApprox(C, opt, x=t)


if __name__ == '__main__':
    mod = Model(np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]), np.array([1, 2, 3]), 15)
    adamOpt = Adam(mod.copy(), h=0.1, tol=1e-5, maxiter=1e4)

    batchFrom = 0
    batchTo = 2
    dataPointsFrom = 0
    dataPointsTo = 500
    train(adamOpt, batchFrom, batchTo, dataPointsFrom, dataPointsTo)

    testBatch = generate_data(batchTo + 1)
    Y0test = np.array([testBatch['Q'][:, dataPointsFrom:dataPointsTo]])
    Ctest = np.array(testBatch['V'][dataPointsFrom:dataPointsTo])
    ttest = testBatch['t'][dataPointsFrom:dataPointsTo]

    testOnNew(Y0test, Ctest, ttest, adamOpt)
