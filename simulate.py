import numpy as np
import matplotlib.pyplot as plt
import scipy
import GPy
import time
import ConfigParser
import io
import time
import os

from factory import ModelFactory, HierarchicalFactory


def gompertz(t,m,A,l):
    return A * np.exp(-np.exp(m*np.e/A*(l-t) + 1))

def generateSample(mu, cov, nugget, length=50):
    noise = np.eye(mu.shape[0])*nugget

    return scipy.stats.multivariate_normal.rvs(mu,cov+noise).reshape((mu.shape[0]/length,length)).T

class Simulation(object):

    @staticmethod
    def search(d='', **kwargs):
        """search directories for config files matching kwargs."""

        for f in os.listdir(d):
            files = os.listdir(os.path.join(d, f))
            configs = filter(lambda x: '.cfg' in x, files)

            if len(configs) == 0:
                continue
            configs = configs[0]

            config = ConfigParser.RawConfigParser()
            config.readfp(io.BytesIO(open(os.path.join(d,f,configs)).read()))

            fail = False

            if config.has_section('main'):
                for k,v in kwargs.iteritems():
                    if config.has_option('main', k):
                        if type(v) == int:
                            if config.getint('main', k) == v:
                                continue
                        elif type(v) == float:
                            if config.getfloat('main', k) == v:
                                continue
                        elif type(v) == str:
                            if config.get('main', k) == v:
                                continue
                    fail = True

            if not fail:
                yield f


    def __init__(self, nobs=15, nbatch = 4, nrep = 3, sigma=.01, batchVariance = 0.05, repVariance = 0.05, mumax=2, A=1, lag=.4, xmax=2, config=None):
        self.nobs = nobs
        self.nbatch = nbatch
        self.nrep = nrep
        self.sigma = sigma
        self.batchVariance = batchVariance
        self.repVariance = repVariance

        self.xmax = xmax

        self.mumax = mumax
        self.A = A
        self.lag = lag

        self.config(config)

        self.setup()

        self._id = int(time.time() * (10**6))

        self.datasets = []
        self.m0 = ModelFactory(self.x0)
        self.m1 = HierarchicalFactory(self.x1,1)
        self.m2 = HierarchicalFactory(self.x2,1)
        self.m3 = HierarchicalFactory(self.x3,2)

    def __repr__(self,):

        return 'simulation-%d-%d-%d-%.5lf-%.5lf-%.5lf' %\
                    (self.nobs, self.nbatch, self.nrep, self.sigma,
                     self.batchVariance, self.repVariance)

    def config(self, cfile):
        if cfile is None:
            return

        config = ConfigParser.RawConfigParser(allow_no_value=True)

        with open(cfile) as f:
            config.readfp(io.BytesIO(f.read()))

        ints = ['nobs', 'nbatch', 'nrep']
        floats = ['sigma', 'batchVariance',
                    'repVariance', 'xmax', 'mumax', 'A', 'lag']

        if config.has_section('main'):
            for i in ints:
                if config.has_option('main', i):
                    self.__dict__[i] = config.getint('main',i)
            for f in floats:
                if config.has_option('main', f):
                    self.__dict__[f] = config.getfloat('main',f)

    def to_config(self, d=''):
        config = ConfigParser.RawConfigParser()
        config.optionxform = str

        config.add_section('main')

        ints = ['nobs', 'nbatch', 'nrep']
        floats = ['sigma', 'batchVariance',
                    'repVariance', 'xmax', 'mumax', 'A', 'lag']

        for i in ints:
            config.set('main', i, self.__dict__[i])
        for f in floats:
            config.set('main', f, self.__dict__[f])

        # config.write(open(os.path.join(d, 'sim_%d.cfg'%self._id), 'w'))
        return config

    def setup(self):
        nobs, nbatch, nrep = self.nobs, self.nbatch, self.nrep
        ntot = nbatch * nrep

        self.x = np.zeros((nobs*ntot,3))

        self.x0 = np.zeros((nobs*ntot,1))
        self.x1 = np.zeros((nobs*ntot,2))
        self.x2 = np.zeros((nobs*ntot,2))

        self.x[:,0] = self.x0[:,0] = self.x1[:,0] = self.x2[:,0] = np.tile(np.linspace(0,self.xmax, nobs), ntot)
        self.x[:,1] = self.x2[:,1] = np.repeat(np.arange(nbatch), nrep*nobs)
        self.x[:,2] = self.x1[:,1] = np.repeat(np.arange(nbatch*nrep), nobs)

        self.x3 = self.x.copy()

        self.xpred = np.zeros((nobs,3))
        self.xpred[:,0] = self.x[:nobs,0]
        self.xpred[:,1:] = -1

        kbatch = GPy.kern.IndependentOutputs(GPy.kern.RBF(1,name='batch', variance=self.batchVariance, lengthscale=.5), index_dim=-2);
        krep = GPy.kern.IndependentOutputs(GPy.kern.RBF(1,name='replicate', variance=self.repVariance, lengthscale=.5))

        n = self.x.shape[0]
        cov = np.zeros((n*4, n*4))

        cov[n:2*n,n:2*n] += krep.K(self.x)
        cov[n:2*n,3*n:] += krep.K(self.x)
        cov[3*n:,n:2*n] += krep.K(self.x)

        cov[2*n:3*n,2*n:3*n] += kbatch.K(self.x)
        cov[2*n:3*n,3*n:] += kbatch.K(self.x)
        cov[3*n:,2*n:3*n] += kbatch.K(self.x)

        cov[3*n:,3*n:] += kbatch.K(self.x) + krep.K(self.x)

        self.sampleCov = cov.copy()

        #self.f = gompertz(self.x[:,0], 2, 1, .4)
        self.f = gompertz(self.x[:,0], self.mumax, self.A, self.lag)

    def generateSamples(self,nsamp=5):

        for _ in range(nsamp):

            s = generateSample(np.tile(self.f, 4), self.sampleCov, self.sigma, self.nobs)

            samp = {}
            samp['mean'] = s.mean()
            samp['std'] = s.std()

            s = (s-s.mean())/s.std()

            samp['sample'] = s

            self.datasets.append(samp)

    def splitSample(self, s):
        return np.array_split(s,4,1)

    def plotSamples(self, s):

        f = (self.f-s['mean'])/s['std']
        s = s['sample']

        diff = s-s.mean(1)[:,None]

        y0, y1 , y2, y3 = self.splitSample(s)

        plt.figure(figsize=(12,6))

        plt.subplot(241)
        plt.plot(self.x[:self.nobs,0], f[:self.nobs],c='k',lw=3)
        for j,z in enumerate(self.x[::self.nobs,1]):
                k = np.unique(self.x[:,1]).tolist().index(z)
                plt.plot(self.x[:self.nobs,0],y0[:,j],color='C%d'%k,alpha=.6);

        plt.subplot(245)
        plt.plot([self.x[:,0].min(),self.x[:,0].max()],[0,0],'k', lw=3)
        for j,z in enumerate(self.x[::self.nobs,1]):
                k = np.unique(self.x[:,1]).tolist().index(z)
                plt.plot(self.x[:self.nobs,0],y0[:,j]-y0.mean(1),color='C%d'%k,alpha=.6);
        plt.ylim(diff.min(), diff.max())

        plt.subplot(242)
        plt.plot(self.x[:self.nobs,0], f[:self.nobs],c='k',lw=3)
        for j,z in enumerate(self.x[::self.nobs,1]):
                k = np.unique(self.x[:,1]).tolist().index(z)
                plt.plot(self.x[:self.nobs,0],y1[:,j],color='C%d'%k,alpha=.6);

        plt.subplot(246)
        plt.plot([self.x[:,0].min(),self.x[:,0].max()],[0,0],'k', lw=3)
        for j,z in enumerate(self.x[::self.nobs,1]):
                k = np.unique(self.x[:,1]).tolist().index(z)
                plt.plot(self.x[:self.nobs,0],y1[:,j]-y1.mean(1),color='C%d'%k,alpha=.6);
        plt.ylim(diff.min(), diff.max())

        plt.subplot(243)
        plt.plot(self.x[:self.nobs,0], f[:self.nobs],c='k',lw=3)
        for j,z in enumerate(self.x[::self.nobs,1]):
                k = np.unique(self.x[:,1]).tolist().index(z)
                plt.plot(self.x[:self.nobs,0],y2[:,j],color='C%d'%k,alpha=.6);

        plt.subplot(247)
        plt.plot([self.x[:,0].min(),self.x[:,0].max()],[0,0],'k', lw=3)
        for j,z in enumerate(self.x[::self.nobs,1]):
                k = np.unique(self.x[:,1]).tolist().index(z)
                plt.plot(self.x[:self.nobs,0],y2[:,j]-y2.mean(1),color='C%d'%k,alpha=.6);
        plt.ylim(diff.min(), diff.max())

        plt.subplot(244)
        plt.plot(self.x[:self.nobs,0], f[:self.nobs],c='k',lw=3)
        for j,z in enumerate(self.x[::self.nobs,1]):
                k = np.unique(self.x[:,1]).tolist().index(z)
                plt.plot(self.x[:self.nobs,0],y3[:,j],color='C%d'%k,alpha=.6);

        plt.subplot(248)
        plt.plot([self.x[:,0].min(),self.x[:,0].max()],[0,0],'k', lw=3)
        for j,z in enumerate(self.x[::self.nobs,1]):
                k = np.unique(self.x[:,1]).tolist().index(z)
                plt.plot(self.x[:self.nobs,0],y3[:,j]-y3.mean(1),color='C%d'%k,alpha=.6);
        plt.ylim(diff.min(), diff.max())
