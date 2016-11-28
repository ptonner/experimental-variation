import ConfigParser, os, scipy
import numpy as np

from gpmultipy.kernel import RBF, White
from gpmultipy import Prior, Model


class Configuration(object):

    def __init__(self,cdir,fname='config.cfg',randomize=False):

        self.config = ConfigParser.ConfigParser()
        self.config.read(os.path.join(cdir,fname))

        self._setDefaults()

        # this is the number of variation levels in the model
        self.levels = self.config.getint('main','levels')

        self._checkLevelConfig()

        self._buildKernels()

        self.p = self.config.getint('main','nrep')
        self.buildDesignMatrix()

        self.x = np.linspace(-1,1)[:,None]
        self.y = np.zeros((self.x.shape[0],self.p))

        if self.config.getboolean('main','uniformPrior'):
            self.sigmaYprior = scipy.stats.uniform(
                                        loc=self.config.getfloat('sigmaYprior','loc'),
                                        scale=self.config.getfloat('sigmaYprior','scale')
                                        )

            self.lengthscalePrior = scipy.stats.uniform(
                                        loc=self.config.getfloat('lengthscalePrior','loc'),
                                        scale=self.config.getfloat('lengthscalePrior','scale'))

            self.sigmaPrior = scipy.stats.uniform(
                                        loc=self.config.getfloat('sigmaPrior','loc'),
                                        scale=self.config.getfloat('sigmaPrior','scale'))
        else:
            self.sigmaYprior = scipy.stats.lognorm(s=self.config.getfloat('sigmaYprior','s'),scale=self.config.getfloat('sigmaYprior','scale'))
            self.lengthscalePrior = scipy.stats.lognorm(s=self.config.getfloat('lengthscalePrior','s'),scale=self.config.getfloat('lengthscalePrior','scale'))
            self.sigmaPrior = scipy.stats.lognorm(s=self.config.getfloat('sigmaPrior','s'),scale=self.config.getfloat('sigmaPrior','scale'))

        self.prior = Prior(self.x,self.k1,[0])
        self.prior2 = Prior(self.x,self.k2,range(1,self.dm.shape[0]))

        if randomize:
            self.model = Model(self.x,self.y,self.dm)

            self.yKernel.sigma = self.sigmaYprior.rvs()
            self.k1.sigma = self.sigmaPrior.rvs()
            self.k1.lengthscale = self.lengthscalePrior.rvs()
            self.k2.sigma = self.sigmaPrior.rvs()
            self.k2.lengthscale = self.lengthscalePrior.rvs()

            self.prior.sample(self.model,self.yKernel)
            self.prior2.sample(self.model,self.yKernel)

    def get(self):
        kernels = [self.yKernel, self.k1]
        kernels += [self.__dict__['k%d'(i+2)] for i in range(self.levels)]

        return self.x, self.p, self.dm, kernels, [self.sigmaYprior, self.lengthscalePrior, self.sigmaPrior], [self.prior, self.prior2]

    def buildDesignMatrix(self):

        self.dm = np.zeros((1+self.p,self.p))
        self.dm[0,:] = 1
        np.fill_diagonal(self.dm[1:,:],1)

    def setDefault(self,name,value):

        if not self.config.has_option("DEFAULT",name):
            self.config.set('DEFAULT',name,value)

    def _setDefaults(self):

        self.setDefault("levels",'1')
        self.setDefault("sigma",'1.0')
        self.setDefault("lengthscale",'1.0')

    def _checkLevelConfig(self):
        """Check that sections are available matching the number of levels, add if needed."""

        for i in range(self.levels):
            if not self.config.has_section("k%d"%(i+2)):
                self.config.add_section("k%d"%(i+2))

    def _buildKernels(self,):
        self.yKernel = White(1,self.config.getfloat('yKernel','sigma'))
        self.k1 = RBF(1,self.config.getfloat('k1','sigma'),self.config.getfloat('k1','lengthscale'))

        for i in range(self.levels):
            k = 'k%d'(i+2)
            self.__dict__[k] = RBF(1,self.config.getfloat(k,'sigma'),self.config.getfloat(k,'lengthscale'))
