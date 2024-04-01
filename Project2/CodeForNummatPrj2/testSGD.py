from allNew import *


if __name__ == '__main__':
    batch = generate_data(0)
    Y0 = batch['P']
    C = batch['T']

    mod = Model(Y0.copy(), C.copy(), 15)
    adamOpt = Adam(mod.copy(), h=0.08, tol=1e-5, maxiter=4*1e3)

    sgdAdamOpt = SGD(adamOpt)
    sgdAdamOpt.separate(3)

    sgdAdamOpt.run()
    print(sgdAdamOpt.M.Jθ)
    plotApprox(sgdAdamOpt)
