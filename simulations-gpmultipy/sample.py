import gpmultipy, scipy, argparse, os
import numpy as np
import pandas as pd

from gpmultipy.kernel import RBF, White, Hierarchical
from gpmultipy.freeze import Freezer
from gpmultipy.sampler.slice import PriorSlice, Slice
from gpmultipy import Model

from config import Configuration
from analysis import Analysis

class Sample(object):

    def __init__(self,config,ds,run,nsample,thin,burnin,levels=-1,randomize=True,randomSamplerOrder=True):

        self.config = Configuration(config,randomizePriors=randomSamplerOrder)

        if randomize:
            self.config.randomize()

        self.run = run
        self.nsample = nsample
        self.thin = thin
        self.burnin = burnin
        self.levels = levels
        self.randomSamplerOrder = randomSamplerOrder

        # default to using all levels
        if self.levels == -1:
            self.levels = self.config.levels

        # how many functions to use
        self.f = sum(self.config.nf[:self.levels+1])

        self.x, self.p, self.dm, self.kernels, self.priors = self.config.get()
        self.yKernel = self.kernels[0]
        for i in range(self.levels+1):
            k = 'k%d'%(i+1)
            self.__dict__[k] = self.kernels[i+1]
        self.kernels = self.kernels[1:self.levels+2]

        if self.config.hierarchy and self.levels < self.config.levels:
            # self.yKernel = self.yKernel.kernels[-1]
            args = self.yKernel.kernels[:self.levels]+[self.yKernel.kernels[-1]]
            self.yKernel = Hierarchical(*args)

        self.y = pd.read_csv(os.path.join(config,ds,'data.csv')).values

        self.dm = self.dm[:self.f,:]

        # self.model = self.config.model
        self.model = Model(self.x,self.y,self.dm)

        if randomize:
            self.model.beta = self.config.model.beta[:,:self.f]

        self._buildSamplers()

        kwargs = {}
        kwargs['yKernel'] = self.yKernel
        for i,k in enumerate(self.kernels):
            kwargs['k%d'%(i+1)] = k

        self.freeze = Freezer(model=self.model,**kwargs)
        self.startSample = self.freeze.freeze()
        self.samples = []

    def load(self,f):
        samples = Analysis.loadSamples(f)
        self.startSample = samples[0]
        self.samples = samples
        self.freeze.push(**self.samples[-1])

    def _buildSamplers(self):
        self.samplers = []

        if self.config.hierarchy:

            self.samplers.append((self.yKernel.__dict__['k%d'%(self.levels+1)],'sigma',
                                    Slice('ySigma',
                                        lambda x: self.model.dataLikelihood(self.yKernel,**{'k%d_sigma'%(self.config.levels+1):x}),
                                        lambda x: self.priors['yKernel']['sigma'].logpdf(x),
                                        self.config.config.getfloat('yKernel','slice-w'),self.config.config.getfloat('yKernel','slice-m'),logspace=True)
                                ))

            for i in range(self.levels):
                self.samplers.append((self.yKernel.__dict__['k%d'%(i+1)],'sigma',
                                        Slice('sigma',
                                            lambda x,i=i: self.model.dataLikelihood(self.yKernel,**{'k%d_sigma'%i:x}),
                                            lambda x: self.priors['k%d'%(i+1)]['sigma'].logpdf(x),
                                            self.config.config.getfloat('k%d'%(i+1),'slice-w'),self.config.config.getfloat('k%d'%(i+1),'slice-m'),logspace=True)
                                    ))

                self.samplers.append((self.yKernel.__dict__['k%d'%(i+1)],'lengthscale',
                                        Slice('lengthscale',
                                            lambda x,i=i: self.model.dataLikelihood(self.yKernel,**{'k%d_lengthscale'%i:x}),
                                            lambda x: self.priors['k%d'%(i+1)]['lengthscale'].logpdf(x),
                                            self.config.config.getfloat('k%d'%(i+1),'slice-w'),self.config.config.getfloat('k%d'%(i+1),'slice-m'),logspace=True)
                                    ))

            self.samplers.append((self.k1,'sigma',
                                    PriorSlice('k1-sigma', self.model,
                                        self.priors['functions'][0], self.priors['k1']['sigma'],
                                        'sigma',
                                        self.config.config.getfloat('k1','slice-w'),
                                        self.config.config.getfloat('k1','slice-m'),
                                        logspace=True)
                                ))

            self.samplers.append((self.k1,'lengthscale',
                                    PriorSlice('k1-lengthscale', self.model,
                                        self.priors['functions'][0], self.priors['k1']['lengthscale'],
                                        'lengthscale',
                                        self.config.config.getfloat('k1','slice-w'),
                                        self.config.config.getfloat('k1','slice-m'),
                                        logspace=True)
                                ))

        else:

            self.samplers.append((self.yKernel,'sigma',
                                    Slice('ySigma',
                                        lambda x: self.model.dataLikelihood(self.yKernel,sigma=x),
                                        lambda x: self.priors['yKernel']['sigma'].logpdf(x),
                                        self.config.config.getfloat('yKernel','slice-w'),self.config.config.getfloat('yKernel','slice-m'),logspace=True)
                                ))

            for i in range(self.levels+1):
                k = 'k%d'%(i+1)

                self.samplers.append((self.__dict__[k],'sigma',
                                        PriorSlice('%s-sigma'%k, self.model,
                                            self.priors['functions'][i], self.priors[k]['sigma'],
                                            'sigma',
                                            self.config.config.getfloat(k,'slice-w'),
                                            self.config.config.getfloat(k,'slice-m'),
                                            logspace=True)
                                    ))

                self.samplers.append((self.__dict__[k],'lengthscale',
                                        PriorSlice('%s-lengthscale'%k, self.model,
                                            self.priors['functions'][i], self.priors[k]['lengthscale'],
                                            'lengthscale',
                                            self.config.config.getfloat(k,'slice-w'),
                                            self.config.config.getfloat(k,'slice-m'),
                                            logspace=True)
                                    ))

    def sample(self):

        for i in range(len(self.samples),self.nsample):
            self._sampleIteration()

            if i % self.thin == 0 and i > self.burnin:
                self.samples.append(self.freeze.freeze())

    def _sampleIteration(self):
        order = range(self.levels+1+len(self.samplers))

        if self.randomSamplerOrder:
            order = np.random.choice(order,len(order),replace=False)

        for o in order:
            if o < self.config.f:
                prior = self.priors['functions'][o]
                prior.sample(self.model,self.yKernel)
            else:
                obj, param, sampler = self.samplers[o-self.levels-1]
                obj.__dict__[param] = sampler.sample(obj.__dict__[param])

        # print

        # for f,prior in self.priors['functions'].iteritems():
        #     if f > self.levels:
        #         continue
        #     prior.sample(self.model,self.yKernel)
        #
        # for obj, param, sampler in self.samplers:
        #     obj.__dict__[param] = sampler.sample(obj.__dict__[param])

    def save(self,dir):
        self.freeze.save(self.samples,os.path.join(dir,'samples.json'))
        self.freeze.save([self.startSample],os.path.join(dir,'startSample.json'))


def main(_type=Sample):

    parser = argparse.ArgumentParser(description='Sample model')

    parser.add_argument('configuration', help='directory containing configuration file')

    parser.add_argument('-l', dest='label', action='store',default='', type=str,
	                   help='label for input')

    parser.add_argument('-r', dest='run', action='store',default='run1', type=str,
	                   help='label for run')

    parser.add_argument('-t',dest='thin',type=int,default=10)
    parser.add_argument('-n',dest='nsample',type=int,default=5000)
    parser.add_argument('-b',dest='burnin',type=int,default=0)
    parser.add_argument('--chain',dest='useChain',action='store_true',help='use existing chain, if possible')
    parser.add_argument('--randomize',dest='randomize',action='store_true',help='randomize starting parameters')
    parser.add_argument('--randomSamplerOrder',dest='randomSamplerOrder',action='store_true',help='randomize parameter sampling order')

    parser.add_argument('-v',dest='levels',type=int,default=-1)

    args = parser.parse_args()

    if not args.run in os.listdir(os.path.join(args.configuration,args.label)):
        os.mkdir(os.path.join(args.configuration,args.label,args.run))

    sample = _type(args.configuration,args.label,args.run,args.nsample,args.thin,args.burnin,args.levels,args.randomize,args.randomSamplerOrder)

    if args.useChain and 'samples.json' in os.listdir(os.path.join(args.configuration,args.label,args.run)):
        sample.load(os.path.join(args.configuration,args.label,args.run,'samples.json'))

    # try:
    #
    # except e2, e:
    #     print e2,e

    sample.sample()

    sample.save(os.path.join(args.configuration,args.label,args.run),)


if __name__ == "__main__":

    main()
