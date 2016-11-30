import os, gpmultipy
from config import Configuration
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Analysis(object):

    @staticmethod
    def loadSamples(path):
        samples = json.load(open(path))
        for s in samples:
            s['model']['beta'] = np.array(s['model']['beta'])
        return samples

    def __init__(self,config):
        self.configDir = config
        self.config = Configuration(config)

        self.datasets = os.listdir(self.configDir)
        self.datasets = filter(lambda x: x[:2]=='ds',self.datasets)

        self.runs = {}
        self.data = {}
        for ds in self.datasets:
            runs = os.listdir(os.path.join(self.configDir,ds))
            runs = filter(lambda x: x[:3]=='run',runs)

            self.runs[ds] = runs
            self.data[ds] = pd.read_csv(os.path.join(self.configDir,ds,'data.csv')).values

    def plotAll(self):
        for ds in self.datasets:
            self.plotDataset(ds)

    def plotDataset(self,ds):
        runs = self.runs[ds]
        data = self.data[ds]

        for r in runs:
            self.plotRun(ds,r)

    def plotRun(self,ds,r,levels=-1,ncol=5,size=4):
        samples = Analysis.loadSamples(os.path.join(self.configDir,ds,r,'samples.json'))

        if levels == -1:
            levels = self.config.levels

        nf = sum(self.config.nf[:levels+1])

        plt.figure(figsize=(size*ncol,size*nf/ncol))
        for f in range(nf):
            plt.subplot(nf/ncol,ncol,f+1)

            fsamples = gpmultipy.ArraySamples(f,samples,'model','beta')
            interval = gpmultipy.interval.FunctionInterval(fsamples,.95)

            interval.plot()
            plt.plot(oracle['model']['beta'][:,f],c='r',alpha=1)

        plt.savefig(os.path.join(self.configDir,'figures',"%s-%s-samples.pdf"%(ds,r)))
