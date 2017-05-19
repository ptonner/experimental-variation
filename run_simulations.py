import matplotlib
matplotlib.use('Agg')

import factory, simulate, scipy, argparse, os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description='Run experimental variation simulation.')

parser.add_argument('config', nargs='?', default=None, help='config file for simulation')

parser.add_argument("--sigma", type=float, default=.01, help='iid noise variance')
parser.add_argument("--batchVariance", type=float, default=0.05, help='variance from batch effects')
parser.add_argument("--repVariance", type=float, default=0.05, help='variance from replicate effects')

parser.add_argument("--nobs", type=int, default=20, help='number of observations for each replicate')
parser.add_argument("--nbatch", type=int, default=2, help='number of batches')
parser.add_argument("--nrep", type=int, default=3, help='number of replicates in each batch')

parser.add_argument("--mumax", type=float, default=2, help='mumax of gompertz model')
parser.add_argument("-A", type=float, default=1, help='carrying capacity of gompertz model')
parser.add_argument("--lag", type=float, default=.3, help='lag of gompertz model')

parser.add_argument("--xmax", type=float, default=2, help='max x value')

parser.add_argument("--nsample", type=int, default=3, help='number of simulations to run')

parser.add_argument("--ntrain", type=int, default=5, help='number of batch GP trainings to run')

args = parser.parse_args()

alpha = .05
thresh = scipy.stats.norm.ppf((1-alpha/2))

sim = simulate.Simulation(
                nobs=args.nobs,
                nrep=args.nrep,
                nbatch=args.nbatch,
                sigma=args.sigma,
                batchVariance=args.batchVariance,
                repVariance=args.repVariance,
                mumax=args.mumax,
                A=args.A,
                lag=args.lag,
                xmax=args.xmax,
                config=args.config
            )

if not str(sim._id) in os.listdir("results/simulations"):
    os.mkdir('results/simulations/%d'%sim._id)

if not 'figures' in os.listdir("results/simulations/%d"%sim._id):
    os.mkdir('results/simulations/%d/figures'%sim._id)

sim.to_config().write(open(os.path.join('results/simulations/%d'%sim._id, 'config.cfg'), 'w'))

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

for k,samp in enumerate(sim.datasets):

    plt.figure(figsize=(10,10))

    ds = samp['sample']

    # models = {}
    for i,y in enumerate(sim.splitSample(ds)):

        ax = plt.subplot2grid((4,5),(i,0))
        plt.plot(y);

        for j, m in enumerate([sim.m0, sim.m1, sim.m2, sim.m3]):
            ax = plt.subplot2grid((4,5),(i,1+j))

            gp,_ = m.batchTrain(y,args.ntrain)

            mu, cov = gp.predict_noiseless(sim.xpred,kern=m.predictionKernel(gp.kern))

            mu = (mu * samp['std']) + samp['mean']
            cov = cov * (samp['std'] ** 2)

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

            f = sim.f.copy()
            #f = (f-samp['mean'])/samp['std']

            accuracy[(i,j)].append(1.*sum((f[:sim.nobs] > mu-thresh*std) & (f[:sim.nobs] < mu+thresh*std))/sim.nobs)
            likelihood[(i,j)].append(gp.log_likelihood())
            interval[(i,j)].append(np.mean(cov[:,0]))

            for key in parameters.keys():
                p = extractParam(gp, *key.split("."))
                if not type(p) == float:
                    p = p.values[0]
                parameters[key][(i,j)].append(p)

            plt.plot(sim.x[:sim.nobs,0], f[:sim.nobs],c='k')

    plt.tight_layout()
    plt.savefig("results/simulations/%d/figures/sim-%d.pdf"%(sim._id, k), bbox_inches='tight')

    pd.DataFrame(accuracy).to_csv("results/simulations/%d/accuracy.csv"%sim._id,index=False)
    pd.DataFrame(likelihood).to_csv("results/simulations/%d/likelihood.csv"%sim._id,index=False)
    pd.DataFrame(interval).to_csv("results/simulations/%d/interval.csv"%sim._id,index=False)
    for key in parameters.keys():
        pd.DataFrame(parameters[key]).to_csv("results/simulations/%d/%s.csv"%(sim._id, key),index=False)
