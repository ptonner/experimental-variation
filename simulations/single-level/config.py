import ConfigParser, os, scipy
import numpy as np

from gpmultipy.kernel import RBF, White
from gpmultipy import Prior

def load(cdir,fname='config.cfg',randomize=False):

    config = ConfigParser.ConfigParser()
    config.read(os.path.join(cdir,fname))

    x = np.linspace(-1,1)[:,None]

    yKernel = White(1,config.getfloat('yKernel','sigma'))
    k1 = RBF(1,config.getfloat('k1','sigma'),config.getfloat('k1','lengthscale'))
    k2 = RBF(1,config.getfloat('k2','sigma'),config.getfloat('k2','lengthscale'))

    if config.getboolean('main','uniformPrior'):
        sigmaYprior = scipy.stats.uniform(
                                    loc=config.getfloat('sigmaYprior','loc'),
                                    scale=config.getfloat('sigmaYprior','scale')
                                    )

        lengthscalePrior = scipy.stats.uniform(
                                    loc=config.getfloat('lengthscalePrior','loc'),
                                    scale=config.getfloat('lengthscalePrior','scale'))

        sigmaPrior = scipy.stats.uniform(
                                    loc=config.getfloat('sigmaPrior','loc'),
                                    scale=config.getfloat('sigmaPrior','scale'))
    else:
        sigmaYprior = scipy.stats.lognorm(s=1.8e0,scale=8e-2)
        lengthscalePrior = scipy.stats.lognorm(s=9e-1,scale=2e0)
        sigmaPrior = scipy.stats.lognorm(s=1.8e0,scale=5e-0)

    if randomize:
        yKernel.sigma = sigmaYprior.rvs()
        k1.sigma = sigmaPrior.rvs()
        k1.lengthscale = lengthscalePrior.rvs()
        k2.sigma = sigmaPrior.rvs()
        k2.lengthscale = lengthscalePrior.rvs()

    p = config.getint('main','nrep')

    dm = np.zeros((1+p,p))
    dm[0,:] = 1
    np.fill_diagonal(dm[1:,:],1)

    prior = Prior(x,k1,[0])
    prior2 = Prior(x,k2,range(1,dm.shape[0]))

    return x, p, dm, [yKernel, k1, k2], [sigmaYprior, lengthscalePrior, sigmaPrior], [prior, prior2]
