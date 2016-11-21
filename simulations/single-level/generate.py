import gpmultipy, scipy, argparse, os, config
import numpy as np
import pandas as pd

from gpmultipy.kernel import RBF, White
from gpmultipy.freeze import Freezer
from gpmultipy import Model

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate data for single-level variation')

    parser.add_argument('configuration', help='directory containing configuration file')

    parser.add_argument('-l', dest='label', action='store',default='', type=str,
	                   help='label for output')

    args = parser.parse_args()

    x, p, dm, kernels, paramPriors, priors = config.load(args.configuration)

    yKernel, k1, k2 = kernels


    if args.label != "" and not args.label in os.listdir(args.configuration):
        os.mkdir(os.path.join(args.configuration,args.label))

    # yKernel.sigma = sigmaYprior.rvs()
    # k1.sigma = sigmaPrior.rvs()
    # k1.lengthscale = lengthscalePrior.rvs()
    # k2.sigma = sigmaPrior.rvs()
    # k2.lengthscale = lengthscalePrior.rvs()

    betaTrue = np.zeros((50,p+1))
    betaTrue[:,0] = scipy.stats.multivariate_normal.rvs(np.zeros(50),k1.K(x))

    for i in range(1,betaTrue.shape[1]):
        betaTrue[:,i] = scipy.stats.multivariate_normal.rvs(np.zeros(50),k2.K(x))

    y = np.dot(betaTrue,dm) + scipy.stats.multivariate_normal.rvs(np.zeros(50),yKernel.K(x),size=p).reshape((50,p))

    model = Model(x,y,dm)
    model.beta = betaTrue

    freeze = Freezer(yKernel=yKernel,k1=k1,k2=k2,model=model)
    trueSample = freeze.freeze()
    freeze.save([trueSample],os.path.join(args.configuration,args.label,'parameters-true.json'))

    pd.DataFrame(y).to_csv(os.path.join(args.configuration,args.label,'data.csv'),index=False)
