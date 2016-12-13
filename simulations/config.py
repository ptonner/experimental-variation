import ConfigParser, os, scipy, patsy
import numpy as np

from gpmultipy.kernel import RBF, White
from gpmultipy import Prior, Model


class Configuration(object):

    def __init__(self,cdir,fname='config.cfg',randomize=False):

        # TODO: seperate nf and ndesigns, which are used incorrectly in non-mean design

        if not os.path.exists(cdir):
            raise ValueError('directory %s does not exist!'%cdir)

        self.config = ConfigParser.ConfigParser()
        self.config.read(os.path.join(cdir,fname))

        self._setDefaults()

        # this is the number of variation levels in the model
        self.levels = self.config.getint('main','levels')

        #experimental design
        self.design = self.config.get('main','design')

        self.intergrateBottom = self.config.getboolean('main','integrate-bottom')

        self.nf = []
        if self.design == 'mean':
            self.nf.append(1)
        elif self.design=='single-treatment':
            self.tmnts = self.config.getint('main','treatments')
            self.nf.append(self.tmnts)
        elif self.design=='multiple-treatment':
            self.tmnts = self.config.getint('main','treatments')
            self.factors = self.config.getint('main','factors')
            self.crossed = self.config.getboolean('main','crossed')

            if self.crossed:
                tot = self.tmnts*self.tmnts*(self.factors-1)
            else:
                tot = self.tmnts*self.factors
            self.nf.append(tot)

        self._checkLevelConfig()

        self._buildKernels()

        # if only one level and nrep provided in main, use that
        if self.levels == 1 and self.config.has_option('main','nrep'):
            self.p = self.config.getint('main','nrep')
            self.nreps = [self.p]
        # otherwise use product of all levels
        else:
            self.nreps = [self.config.getint('level%d'%(i+1),'nrep') for i in range(self.levels)]
            self.p = np.prod(self.nreps)

        # don't build functions/priors for bottom level if integrating out
        if self.intergrateBottom:
            stahp = self.levels
        else:
            stahp = self.levels+1

        # add the number of functions from replicate structure
        self.nf += [self.nf[0]*np.product(self.nreps[:i]) for i in range(1,stahp)]
        self.cumnf = np.cumsum(self.nf)
        self.cumnreps = np.cumprod(self.nreps)
        self.f = sum(self.nf)

        self.buildDesignMatrix()

        self.x = np.linspace(self.config.getfloat('main','xmin'),
                             self.config.getfloat('main','xmax'),
                             self.config.getint('main','n'))[:,None]
                             
        self.y = np.zeros((self.x.shape[0],self.dm.shape[1]))

        self.priors = {'yKernel':{}, 'functions':{}, 'k1':{}}
        for i in range(self.levels+1):
            self.priors['k%d'%(i+1)] = {}

        if self.config.getboolean('main','uniformPrior'):
            self.priors['yKernel']['sigma'] = scipy.stats.uniform(
                                        loc=self.config.getfloat('sigmaYprior','loc'),
                                        scale=self.config.getfloat('sigmaYprior','scale')
                                        )

            for i in range(self.levels+1):
                self.priors['k%d'%(i+1)]['lengthscale'] = scipy.stats.uniform(
                                            loc=self.config.getfloat('lengthscalePrior','loc'),
                                            scale=self.config.getfloat('lengthscalePrior','scale'))

                self.priors['k%d'%(i+1)]['sigma'] = scipy.stats.uniform(
                                            loc=self.config.getfloat('sigmaPrior','loc'),
                                            scale=self.config.getfloat('sigmaPrior','scale'))
        else:
            self.priors['yKernel']['sigma'] = scipy.stats.lognorm(s=self.config.getfloat('sigmaYprior','s'),scale=self.config.getfloat('sigmaYprior','scale'))

            for i in range(self.levels+1):
                self.priors['k%d'%(i+1)]['lengthscale'] = scipy.stats.lognorm(s=self.config.getfloat('lengthscalePrior','s'),scale=self.config.getfloat('lengthscalePrior','scale'))
                self.priors['k%d'%(i+1)]['priors'] = scipy.stats.lognorm(s=self.config.getfloat('sigmaPrior','s'),scale=self.config.getfloat('sigmaPrior','scale'))

        # self.priors['functions'][0] = Prior(self.x,self.k1,[0])

        for i in range(stahp):
            start = sum(self.nf[:i])
            stahp2 = start+self.nf[i]

            self.priors['functions'][i] = Prior(self.x,self.__dict__['k%d'%(i+1)],range(start,stahp2))

    def get(self):
        kernels = [self.yKernel, self.k1]
        kernels += [self.__dict__['k%d'%(i+2)] for i in range(self.levels)]

        return self.x, self.dm.shape[1], self.dm, kernels, self.priors

    def randomize(self):
        self.model = Model(self.x,self.y,self.dm)

        for f,prior in self.priors['functions'].iteritems():
            prior.sample(self.model,self.yKernel)

        self.yKernel.sigma = self.priors['yKernel']['sigma'].rvs()

        for i in range(self.levels+1):
            k = "k%d"%(i+1)
            for param in self.priors[k]:
                self.__dict__[k].__dict__[param] = self.priors[k][param].rvs()



    def buildDesignMatrix(self):

        # designs: (f,d) f functions by d designs
        self.designs = None
        if self.design == 'mean':
            self.designs = np.ones((1,1))
        elif self.design == 'single-treatment':
            self.contrast = patsy.contrasts.Helmert().code_without_intercept(range(self.tmnts)).matrix
            self.designs = np.row_stack((np.ones((1,self.contrast.shape[0])), self.contrast.T))

        k = 0
        self.replicates = np.zeros((np.sum(self.cumnreps),self.p))
        # self.replicates = np.zeros((100,self.p))
        for i in range(self.levels):
            stab = max(np.prod(self.nreps[i+1:]),1) # step-size
            stab = int(stab)

            # print i,stab

            # for j in range(self.nreps[i]):
            for j in range(self.p/stab):
                # print i,j,k
                self.replicates[k,j*stab:(j+1)*stab] = 1
                k+=1

        self.dm = np.zeros((self.f,self.p*self.nf[0]))
        self.dm[:self.nf[0],:] = self.designs.repeat(self.p,1)

        offset = self.nf[0]
        for i in range(self.nf[0]):
            col = i*self.p
            row = i*self.replicates.shape[0] + offset

            self.dm[row:row+self.replicates.shape[0],col:col+self.replicates.shape[1]] = self.replicates

    def setDefault(self,name,value):

        if not self.config.has_option("DEFAULT",name):
            self.config.set('DEFAULT',name,value)

    def _setDefaults(self):

        self.setDefault("levels",'1')
        self.setDefault("n",'50')
        self.setDefault("design",'mean')
        self.setDefault("treatments",'2')
        self.setDefault("factors",'2')
        self.setDefault("crossed",'False')
        self.setDefault("variable-selection",'False')
        self.setDefault("sigma",'1.0')
        self.setDefault("lengthscale",'1.0')
        self.setDefault("nrep",'3')
        self.setDefault('uniformPrior','True')
        self.setDefault('loc','1.0')
        self.setDefault('scale','1.0')
        self.setDefault('s','1.0')
        self.setDefault('xmin','-1')
        self.setDefault('xmax','1')
        self.setDefault('slice-w','.2')
        self.setDefault('slice-m','5')

        # toggle whether to integrate out the lowest level of hierarchy
        self.setDefault('integrate-bottom','False')

    def _checkLevelConfig(self):
        """Check that sections are available matching the number of levels, add if needed."""

        for i in range(self.levels):

            # check for kernels for each level
            if not self.config.has_section("k%d"%(i+2)):
                self.config.add_section("k%d"%(i+2))

            # check for level config section
            if not self.config.has_section("level%d"%(i+1)):
                self.config.add_section("level%d"%(i+1))

    def _buildKernels(self,):
        self.yKernel = White(1,self.config.getfloat('yKernel','sigma'))
        self.k1 = RBF(1,self.config.getfloat('k1','sigma'),self.config.getfloat('k1','lengthscale'))

        for i in range(self.levels):
            k = 'k%d'%(i+2)
            self.__dict__[k] = RBF(1,self.config.getfloat(k,'sigma'),self.config.getfloat(k,'lengthscale'))
