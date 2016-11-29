import gpmultipy, scipy, argparse, os
import numpy as np
import pandas as pd

from gpmultipy.kernel import RBF, White
from gpmultipy.freeze import Freezer
from gpmultipy.sampler.slice import Slice
from gpmultipy import Model

from config import Configuration

class Sample(object):

    def __init__(self,config,ds,run,nsample,thin,burnin):
        self.config = Configuration(config)
        self.config.randomize()
        self.run = run
        self.nsample = nsample
        self.thin = thin
        self.burnin = burnin

        self.x, self.p, self.dm, self.kernels, self.priors = self.config.get()
        self.yKernel = self.kernels[0]
        for i in range(self.config.levels+1):
            k = 'k%d'%(i+1)
            self.__dict__[k] = self.kernels[i+1]

        self.y = pd.read_csv(os.path.join(config,ds,'data.csv')).values

        self.model = Model(self.x,self.y,self.dm)

        self._buildSamplers()

        kwargs = {}
        kwargs['yKernel'] = self.yKernel
        for i,k in enumerate(self.kernels):
            kwargs['k%d'%(i+1)] = k

        self.freeze = Freezer(model=self.model,**kwargs)
        self.startSample = self.freeze.freeze()
        self.samples = []

    def _buildSamplers(self):
        self.samplers = []

        self.samplers.append((self.yKernel,'sigma',
                                Slice('ySigma',
                                    lambda x: self.model.dataLikelihood(self.yKernel,sigma=x),
                                    lambda x: self.priors['yKernel']['sigma'].logpdf(x),
                                    .2,5,logspace=True)
                            ))

        for i in range(self.config.levels+1):
            k = 'k%d'%(i+1)

            self.samplers.append((self.__dict__[k],'sigma',
                                    Slice('%s-sigma'%k,
                                        lambda x: self.priors['functions'][i].loglikelihood(self.model.beta,sigma=x),
                                        lambda x: self.priors[k]['sigma'].logpdf(x),
                                        .2,5,logspace=True)
                                ))

            self.samplers.append((self.__dict__[k],'lengthscale',
                                    Slice('%s-lengthscale'%k,
                                        lambda x: self.priors['functions'][i].loglikelihood(self.model.beta,lengthscale=x),
                                        lambda x: self.priors[k]['lengthscale'].logpdf(x),
                                        .2,5,logspace=True)
                                ))

    def sample(self):

        for i in range(self.nsample):
            self._sampleIteration()

            if i % self.thin == 0 and i > self.burnin:
                self.samples.append(self.freeze.freeze())

    def _sampleIteration(self):

        for f,prior in self.priors['functions'].iteritems():
            prior.sample(self.model,self.yKernel)

        for obj, param, sampler in self.samplers:
            obj.__dict__[param] = sampler.sample(obj.__dict__[param])

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

    args = parser.parse_args()

    if not args.run in os.listdir(os.path.join(args.configuration,args.label)):
        os.mkdir(os.path.join(args.configuration,args.label,args.run))

    sample = _type(args.configuration,args.label,args.run,args.nsample,args.thin,args.burnin)
    sample.sample()
    sample.save(os.path.join(args.configuration,args.label,args.run),)


if __name__ == "__main__":

    main()