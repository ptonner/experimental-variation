import os, gpmultipy, json, sys
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
        self.datasets.sort()

        self.runs = {}
        self.data = {}
        for ds in self.datasets:
            runs = os.listdir(os.path.join(self.configDir,ds))
            runs = filter(lambda x: x[:3]=='run',runs)
            runs.sort()

            self.runs[ds] = runs
            self.data[ds] = pd.read_csv(os.path.join(self.configDir,ds,'data.csv')).values

    def loadDatasetRun(self,ds,r):
        if not self.checkRun(ds,r):
            return None,None

        samples = Analysis.loadSamples(os.path.join(self.configDir,ds,r,'samples.json'))
        oracle = Analysis.loadSamples(os.path.join(self.configDir,ds,'parameters-true.json'))[0]

        return samples,oracle

    def check(self,):
        for ds in self.datasets:
            # print ds
            sys.stdout.write('%s' % ds)

            names,checks = self.checkDataset(ds)
            checks = np.array(checks)
            checks = pd.DataFrame(checks,index=names)
            checks.to_csv(os.path.join(self.configDir,ds,'checks.csv'),index=True,)

            print

    def checkDataset(self,ds):
        runs = self.runs[ds]
        data = self.data[ds]

        checks = []
        names = []
        for r in runs:
            sys.stdout.write('.'); sys.stdout.flush()
            runcheck  = self.checkDatasetRun(ds,r)

            if runcheck is None:
                continue

            if len(runcheck) < self.config.f:
                runcheck += [np.nan] * (self.config.f-len(runcheck))

            checks.append(runcheck)
            names.append(r)

        return names,checks

    def checkDatasetRun(self,ds,r):
        samples,oracle = self.loadDatasetRun(ds,r)

        if samples is None or oracle is None:
            return

        nf = samples[0]['model']['beta'].shape[1]

        chex = []
        for f in range(nf):
            fsamples = gpmultipy.ArraySamples(f,samples,'model','beta')
            interval = gpmultipy.interval.FunctionInterval(fsamples,.95)

            chex.append(int(interval.contains(oracle['model']['beta'][:,f])))

        return chex

    def plotAll(self):
        for ds in self.datasets:
            self.plotDataset(ds)

    def plotDataset(self,ds):
        runs = self.runs[ds]
        data = self.data[ds]

        for r in runs:
            print ds,r

            if 'noHierarchy' in r:
                pass
                # self.plotRun(ds,r,0)
            else:
                self.plotRun(ds,r)

                break # REMOVE

    def plotRun(self,ds,r,levels=-1,ncol=5,size=4):
        if not self.checkRun(ds,r):
            return

        samples = Analysis.loadSamples(os.path.join(self.configDir,ds,r,'samples.json'))
        oracle = Analysis.loadSamples(os.path.join(self.configDir,ds,'parameters-true.json'))[0]

        # self.plotFunctions(samples,oracle)
        # plt.savefig(os.path.join(self.configDir,'figures',"%s-%s-samples.pdf"%(ds,r)))
        # plt.close()

        for di in range(self.config.designs.shape[0]):
            self.plotDesignReplication(samples,oracle,di)

            plt.savefig(os.path.join(self.configDir,'figures',"%s-%s-design%d.pdf"%(ds,r,di)))
            plt.close()

    def plotFunctions(self,samples,oracle,ncol=5,size=4):
        if len(samples) == 0:
            return

        if levels == -1:
            levels = self.config.levels

        nf = sum(self.config.nf[:levels+1])
        offset = int(nf%ncol>0)

        if nf < ncol:
            ncol = nf

        plt.figure(figsize=(size*ncol,size*(nf/ncol + offset)))
        for f in range(nf):
            plt.subplot(nf/ncol+offset,ncol,f+1)

            fsamples = gpmultipy.ArraySamples(f,samples,'model','beta')
            interval = gpmultipy.interval.FunctionInterval(fsamples,.95)

            interval.plot()
            plt.plot(oracle['model']['beta'][:,f],c='r',alpha=1)

    def plotDesignReplication(self,samples,oracle,designInd,levels=-1):
        """Plot the replication functions of a design"""

        if levels == -1:
            levels = self.config.levels

        if levels == 0:
            return

        step = self.config.replicates.shape[0] # size of each replicate block
        ind = step*designInd # where this design's replicates start
        offset = self.config.nf[0] # number of controlled functions

        # these are the replicate fxns for this design
        fxns = range(offset+ind*step,offset+(ind+1)*step)

        print designInd, self.config.nf, self.config.designs.shape, self.config.dm.shape, self.config.replicates.shape,fxns,ind

        # max width of figure
        width = self.config.cumnreps[levels-1]
        height = levels + 1
        size = 4

        plt.figure(figsize=(size*width,size*height))

        plt.subplot2grid((height,width),(0,0),colspan=width)
        fsamples = gpmultipy.ArraySamples(designInd,samples,'model','beta')
        interval = gpmultipy.interval.FunctionInterval(fsamples,.95)

        interval.plot()
        plt.plot(oracle['model']['beta'][:,designInd],c='r',alpha=1)

        k = 0
        for i in range(levels):
            levelwidth = width/self.config.cumnreps[i]

            for j in range(self.config.cumnreps[i]):
                plt.subplot2grid((height,width),(i+1,j*levelwidth),colspan=levelwidth)

                find = fxns[k]
                k += 1

                fsamples = gpmultipy.ArraySamples(find,samples,'model','beta')
                interval = gpmultipy.interval.FunctionInterval(fsamples,.95)

                interval.plot()
                plt.plot(oracle['model']['beta'][:,find],c='r',alpha=1)
        print 'finished'

    def checkRun(self,ds,r):
        return 'samples.json' in os.listdir(os.path.join(self.configDir,ds,r,))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Analyze results of sampling model')

    parser.add_argument('configuration', help='directory containing configuration file')
    parser.add_argument('-p',dest="plot",action="store_true",default=False,help='plot results')
    parser.add_argument('-c',dest="check",action="store_true",default=False,help='check intervals')

    args = parser.parse_args()

    # print args.plot,args.check

    analysis = Analysis(args.configuration)

    if args.plot:
        analysis.plotAll()
    if args.check:
        analysis.check()
