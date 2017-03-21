from sample import Sample, main
from gpmultipy.kernel import Addition
from gpmultipy.sampler.slice import PriorSlice,Slice
from gpmultipy.freeze import Freezer

import numpy as np


class SampleIntegrate(Sample):
    """Integrate out the lowest level of hierarchical variation."""

    def __init__(self,config,ds,run,nsample,thin,burnin,levels=-1,*args,**kwargs):
        Sample.__init__(self,config,ds,run,nsample,thin,burnin,levels,*args,**kwargs)
        self.levels=self.config.levels-1

        kern = self.__dict__['k%d'%(self.levels+2)]
        self.yKernelOld = self.yKernel
        self.yKernel = Addition(self.yKernel,kern)

        del self.kernels[-1]

        kwargs = {}
        kwargs['yKernel'] = self.yKernel
        for i,k in enumerate(self.kernels):
            kwargs['k%d'%(i+1)] = k

        self.freeze = Freezer(model=self.model,**kwargs)
        self.startSample = self.freeze.freeze()

    def _buildSamplers(self):
        self.samplers = []

        self.samplers.append((self.yKernel,'k1_sigma',
                                Slice('ySigma',
                                    lambda x: self.model.dataLikelihood(self.yKernel,k1_sigma=x),
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

        k = 'k%d'%(self.levels+1)
        self.samplers.append((self.yKernel,'k2_sigma',
                                Slice('ySigma2',
                                    lambda x: self.model.dataLikelihood(self.yKernel,k2_sigma=x),
                                    lambda x: self.priors[k]['sigma'].logpdf(x),
                                    self.config.config.getfloat(k,'slice-w'),self.config.config.getfloat(k,'slice-m'),logspace=True)
                            ))
        self.samplers.append((self.yKernel,'k2_lengthscale',
                                Slice('yLengthscale2',
                                    lambda x: self.model.dataLikelihood(self.yKernel,k2_lengthscale=x),
                                    lambda x: self.priors[k]['lengthscale'].logpdf(x),
                                    self.config.config.getfloat(k,'slice-w'),self.config.config.getfloat(k,'slice-m'),logspace=True)
                            ))

    def _sampleIteration(self):
        order = range(self.levels+1+len(self.samplers))

        if self.randomSamplerOrder:
            order = np.random.choice(order,len(order),replace=False)

        for o in order:
            if o <= self.levels:
                prior = self.priors['functions'][o]
                prior.sample(self.model,self.yKernel)
            else:
                obj, param, sampler = self.samplers[o-self.levels-1]
                # print param, obj

                if obj == self.yKernelOld:
                    obj = self.yKernel
                    if param == 'k1_sigma':
                        obj.__dict__['k1'].sigma = sampler.sample(obj.__dict__['k1'].sigma)
                    elif param == 'k2_sigma':
                        obj.__dict__['k2'].sigma = sampler.sample(obj.__dict__['k2'].sigma)
                    elif param == 'k2_lengthscale':
                        obj.__dict__['k2'].lengthscale = sampler.sample(obj.__dict__['k2'].lengthscale)
                    else:
                        raise ValueError("unknown parameter %s!"%param)
                else:
                    obj.__dict__[param] = sampler.sample(obj.__dict__[param])

if __name__ == "__main__":
    main(SampleIntegrate)
