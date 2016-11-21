import gpmultipy, scipy, argparse, os
import numpy as np

from gpmultipy import dataset
from gpmultipy.model import Model
from gpmultipy.prior import Prior
from gpmultipy.kernel import RBF, White
from gpmultipy.freeze import Freezer
from gpmultipy.sampler.slice import Slice
from gpmultipy.kernel import Product, Linear

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run simulation of single layer hierarchy')

    parser.add_argument('-l', dest='label', action='store',default='', type=str,
	                   help='label for output')

    parser.add_argument('-d', dest='odir', action='store',default='', type=str,
	                   help='output directory')

    parser.add_argument('-u', dest='uniformPrior', action='store_true',
	                   help='use uniform priors')

    args = parser.parse_args()

    x = np.linspace(-1,1)[:,None]

    yKernel = White(1,.001)
    k1 = RBF(1,1,1)
    k2 = RBF(1,.05,.3)

    p = 10
    betaTrue = np.zeros((50,p+1))
    betaTrue[:,0] = scipy.stats.multivariate_normal.rvs(np.zeros(50),k1.K(x))

    for i in range(1,betaTrue.shape[1]):
        betaTrue[:,i] = scipy.stats.multivariate_normal.rvs(np.zeros(50),k2.K(x))

    dm = np.zeros((1+p,p))
    dm[0,:] = 1
    np.fill_diagonal(dm[1:,:],1)

    prior = Prior(x,k1,[0])
    prior2 = Prior(x,k2,range(1,dm.shape[0]))

    y = np.dot(betaTrue,dm) + scipy.stats.norm.rvs(0,np.sqrt(yKernel.sigma),size=50*p).reshape((50,p))

    model = Model(x,y,dm)
    model.beta = betaTrue

    if args.uniformPrior:
        print 'using uniform prior'
        sigmaYprior = scipy.stats.uniform(loc=1e-5,scale=1)
        lengthscalePrior = scipy.stats.uniform(loc=1e-2,scale=1e2)
        sigmaPrior = scipy.stats.uniform(loc=1e-2,scale=1e2)
    else:
        sigmaYprior = scipy.stats.lognorm(s=1.8e0,scale=8e-2)
        lengthscalePrior = scipy.stats.lognorm(s=9e-1,scale=2e0)
        sigmaPrior = scipy.stats.lognorm(s=1.8e0,scale=5e-0)

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

    samples = []
    freeze = Freezer(yKernel=yKernel,k1=k1,k2=k2,model=model)
    trueSample = freeze.freeze()


    """
    RANDOM-EYES
    """
    yKernel.sigma = sigmaYprior.rvs()
    k1.sigma = sigmaPrior.rvs()
    k1.lengthscale = lengthscalePrior.rvs()
    k2.sigma = sigmaPrior.rvs()
    k2.lengthscale = lengthscalePrior.rvs()
    #
    # model.beta[:,0] = scipy.stats.multivariate_normal.rvs(np.zeros(50),k1.K(x))
    # for i in range(1,model.beta.shape[1]):
    #     model.beta[:,i] = scipy.stats.multivariate_normal.rvs(np.zeros(50),k2.K(x))
    prior.sample(model,yKernel)
    prior2.sample(model,yKernel)

    f= freeze.freeze()
    del f['model']
    print f

    """
    SAMPLE
    """
    thin = 10
    burnin = 0
    nsample = 5000

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
    freeze.save(samples,os.path.join(args.odir,'single-level%s-samples.json'%label))
    freeze.save([trueSample],os.path.join(args.odir,'single-level%s-true.json'%label))
