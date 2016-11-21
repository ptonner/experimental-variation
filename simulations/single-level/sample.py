import gpmultipy, scipy, argparse, os, config
import numpy as np
import pandas as pd

from gpmultipy.kernel import RBF, White
from gpmultipy.freeze import Freezer
from gpmultipy.sampler.slice import Slice
from gpmultipy import Model

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Sample model for data with single-level variation')

    parser.add_argument('configuration', help='directory containing configuration file')

    parser.add_argument('-l', dest='label', action='store',default='', type=str,
	                   help='label for output')

    parser.add_argument('-r', dest='run', action='store',default='run1', type=str,
	                   help='label for run')

    parser.add_argument('-t',dest='thin',type=int,default=10)
    parser.add_argument('-n',dest='nsample',type=int,default=5000)
    parser.add_argument('-b',dest='burnin',type=int,default=200)

    args = parser.parse_args()

    if not args.run in os.listdir(os.path.join(args.configuration,args.label)):
        os.mkdir(os.path.join(args.configuration,args.label,args.run))

    x, p, dm, kernels, paramPriors, priors = config.load(args.configuration)
    yKernel, k1, k2 = kernels
    sigmaYprior, lengthscalePrior, sigmaPrior = paramPriors
    prior, prior2 = priors

    y = pd.read_csv(os.path.join(args.configuration,args.label,'data.csv')).values
    model = Model(x,y,dm)

    # randomize

    yKernel.sigma = sigmaYprior.rvs()
    k1.sigma = sigmaPrior.rvs()
    k1.lengthscale = lengthscalePrior.rvs()
    k2.sigma = sigmaPrior.rvs()
    k2.lengthscale = lengthscalePrior.rvs()

    # model.beta[:,0] = scipy.stats.multivariate_normal.rvs(np.zeros(50),k1.K(x))
    # for i in range(1,model.beta.shape[1]):
    #     model.beta[:,i] = scipy.stats.multivariate_normal.rvs(np.zeros(50),k2.K(x))
    prior.sample(model,yKernel)
    prior2.sample(model,yKernel)

    """
    SAMPLE
    """
    ySigmaSlice = Slice('ySigma',
                        lambda x: model.dataLikelihood(yKernel,sigma=x),
                        lambda x: sigmaYprior.logpdf(x),
                        .2,5,logspace=True)

    kSigmaSlice = Slice('kSigma',
                        lambda x: prior.loglikelihood(model.beta,sigma=x),
                        lambda x: sigmaPrior.logpdf(x),
                        .2,5,logspace=True)

    kLengthscaleSlice = Slice('kLengthscale',
                        lambda x: prior.loglikelihood(model.beta,lengthscale=x),
                        lambda x: lengthscalePrior.logpdf(x),
                        .2,5,logspace=True)

    k2SigmaSlice = Slice('kSigma',
                        lambda x: prior2.loglikelihood(model.beta,sigma=x),
                        lambda x: sigmaPrior.logpdf(x),
                        .2,5,logspace=True)

    k2LengthscaleSlice = Slice('kLengthscale',
                        lambda x: prior2.loglikelihood(model.beta,lengthscale=x),
                        lambda x: lengthscalePrior.logpdf(x),
                        .2,5,logspace=True)


    thin = args.thin
    burnin = args.burnin
    nsample = args.nsample
    samples = []
    freeze = Freezer(yKernel=yKernel,k1=k1,k2=k2,model=model)
    startSample = freeze.freeze()

    for i in range(nsample):
        prior.sample(model,yKernel)
        prior2.sample(model,yKernel)

        yKernel.sigma = ySigmaSlice.sample(yKernel.sigma)

        k1.sigma = kSigmaSlice.sample(k1.sigma)
        k1.lengthscale = kLengthscaleSlice.sample(k1.lengthscale)

        k2.sigma = k2SigmaSlice.sample(k2.sigma)
        k2.lengthscale = k2LengthscaleSlice.sample(k2.lengthscale)

        if i % thin == 0 and i > burnin:
            samples.append(freeze.freeze())

    label=""
    if args.label != "":
        label = "-"+args.label
    freeze.save(samples,os.path.join(args.configuration,args.label,args.run,'samples.json'))
    freeze.save([startSample],os.path.join(args.configuration,args.label,args.run,'startSample.json'))
