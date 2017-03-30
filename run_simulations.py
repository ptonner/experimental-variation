import matplotlib
matplotlib.use('Agg')

import factory, simulate, scipy, argparse, os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description='Run experimental variation simulation.')

parser.add_argument("--sigma", type=float, default=.01)
parser.add_argument("--batchVariance", type=float, default=0.005)
parser.add_argument("--repVariance", type=float, default=0.005)

parser.add_argument("--nobs", type=float, default=15)
parser.add_argument("--nbatch", type=float, default=4)
parser.add_argument("--nrep", type=int, default=3)

parser.add_argument("--nsample", type=int, default=3)

args = parser.parse_args()

alpha = .05
thresh = scipy.stats.norm.ppf((1-alpha/2))

sim = simulate.Simulation(
                nobs=args.nobs,
                nrep=args.nrep,
                nbatch=args.nbatch,
                sigma=args.sigma,
                batchVariance=args.batchVariance,
                repVariance=args.repVariance
            )

if not str(sim) in os.listdir("results/simulations"):
    os.mkdir('results/simulations/%s'%str(sim))

sim.generateSamples(args.nsample)

interval = {}
accuracy = {}
likelihood = {}
parameters = {
                'likelihood.variance': {},
                'kern.base.variance':{},
                'kern.level0.variance':{},
                'kern.level1.variance':{}
             }

def extractParam(m, *args):
    if len(args) == 0:
        return m
    else:
        if args[0] in m.__dict__:
            return extractParam(m.__dict__[args[0]], *args[1:])
        else:
            return np.nan

for k,ds in enumerate(sim.datasets):

    plt.figure(figsize=(10,10))

    # models = {}
    for i,y in enumerate(sim.splitSample(ds)):

        ax = plt.subplot2grid((4,5),(i,0))
        plt.plot(y);

        for j, m in enumerate([sim.m0, sim.m1, sim.m2, sim.m3]):
            ax = plt.subplot2grid((4,5),(i,1+j))

            gp,_ = m.batchTrain(y,5)

            # models[(i,j)] = gp

            mu, cov = gp.predict_noiseless(sim.xpred,kern=m.predictionKernel(gp.kern))
            mu = mu[:,0]
            std = np.sqrt(cov[:,0])

            plt.plot(sim.xpred[:,0], mu)
            plt.fill_between(sim.xpred[:,0], mu-2*std,mu+2*std, alpha=.4)

            if not (i,j) in interval:
                interval[(i,j)] = []
                accuracy[(i,j)] = []
                likelihood[(i,j)] = []

                for key in parameters.keys():
                    parameters[key][(i,j)] = []

            # if all((sim.f[:sim.nobs] > mu-thresh*std) & (sim.f[:sim.nobs] < mu+thresh*std)):
            #     intervals[(i,j)] += 1
            accuracy[(i,j)].append(1.*sum((sim.f[:sim.nobs] > mu-thresh*std) & (sim.f[:sim.nobs] < mu+thresh*std))/sim.nobs)
            likelihood[(i,j)].append(gp.log_likelihood())
            interval[(i,j)].append(np.mean(cov[:,0]))

            for key in parameters.keys():
                p = extractParam(gp, *key.split("."))
                if not type(p) == float:
                    p = p.values[0]
                parameters[key][(i,j)].append(p)

            plt.plot(sim.x[:sim.nobs,0], sim.f[:sim.nobs],c='k')

    plt.tight_layout()
    plt.savefig("results/simulations/%s/sim-%d.pdf"%(str(sim), k), bbox_inches='tight')

    pd.DataFrame(accuracy).to_csv("results/%s-accuracy.csv"%str(sim),index=False)
    pd.DataFrame(likelihood).to_csv("results/%s-likelihood.csv"%str(sim),index=False)
    pd.DataFrame(interval).to_csv("results/%s-interval.csv"%str(sim),index=False)
    for key in parameters.keys():
        pd.DataFrame(parameters[key]).to_csv("results/%s-%s.csv"%(str(sim), key),index=False)

# for k in intervals.keys():
#     intervals[k] = 1.*intervals[k]/len(sim.datasets)

# pd.DataFrame(accuracy).to_csv("results/sim-accuracy.csv",index=False)
