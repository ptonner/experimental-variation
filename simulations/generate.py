import gpmultipy, scipy, argparse, os
import numpy as np
import pandas as pd

from config import Configuration
from gpmultipy.kernel import RBF, White
from gpmultipy.freeze import Freezer
from gpmultipy import Model

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate data for single-level variation')

    parser.add_argument('configuration', help='directory containing configuration file')

    parser.add_argument('-l', dest='label', action='store',default='', type=str,
	                   help='label for output')

    parser.add_argument('-f', dest='fname', action='store',default='config.cfg', type=str,
	                   help='config filename')

    args = parser.parse_args()

    config = Configuration(args.configuration,args.fname)
    x, p, dm, kernels, priors = config.get()

    if args.label != "" and not args.label in os.listdir(args.configuration):
        os.mkdir(os.path.join(args.configuration,args.label))

    if 'data.csv' in os.path.join(args.configuration,args.label):
        raise ValueError("dataset already exists at %s!"%
                            os.path.join(args.configuration,args.label))

    betaTrue = np.zeros((x.shape[0],dm.shape[0]))

    for f in range(config.f):
        ind = min(np.where(f < config.cumnf)[0]) + 1
        # print f,ind

        kernel = kernels[ind]
        betaTrue[:,f] = scipy.stats.multivariate_normal.rvs(np.zeros(x.shape[0]),kernel.K(x))

    y = np.dot(betaTrue,dm) + scipy.stats.multivariate_normal.rvs(np.zeros(x.shape[0]),kernels[0].K(x),size=p).reshape((x.shape[0],p))

    model = Model(x,y,dm)
    model.beta = betaTrue

    kwargs = {'k%d'%(i+1):kernels[i+1] for i in range(config.levels+1)}
    freeze = Freezer(yKernel=kernels[0],model=model,**kwargs)
    trueSample = freeze.freeze()
    freeze.save([trueSample],os.path.join(args.configuration,args.label,'parameters-true.json'))

    pd.DataFrame(y).to_csv(os.path.join(args.configuration,args.label,'data.csv'),index=False)
