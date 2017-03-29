import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import factory, simulate, scipy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

alpha = .05
thresh = scipy.stats.norm.ppf((1-alpha/2))

sim = simulate.Simulation()
sim.generateSamples(50)

intervals = {}
accuracy = {}

for k,ds in enumerate(sim.datasets):

    plt.figure(figsize=(10,10))

    models = {}
    for i,y in enumerate(sim.splitSample(ds)):

        ax = plt.subplot2grid((4,5),(i,0))
        plt.plot(y);

        for j, m in enumerate([sim.m0, sim.m1, sim.m2, sim.m3]):
            ax = plt.subplot2grid((4,5),(i,1+j))

            gp,_ = m.batchTrain(y)

            models[(i,j)] = gp

            mu, cov = gp.predict_noiseless(sim.xpred,kern=m.predictionKernel(gp.kern))
            mu = mu[:,0]
            std = np.sqrt(cov[:,0])

            plt.plot(sim.xpred[:,0], mu)
            plt.fill_between(sim.xpred[:,0], mu-2*std,mu+2*std, alpha=.4)

            if not (i,j) in intervals:
                intervals[(i,j)] = 0
                accuracy[(i,j)] = []

            if all((sim.f[:sim.nobs] > mu-thresh*std) & (sim.f[:sim.nobs] < mu+thresh*std)):
                intervals[(i,j)] += 1
            accuracy[(i,j)].append(1.*sum((sim.f[:sim.nobs] > mu-thresh*std) & (sim.f[:sim.nobs] < mu+thresh*std))/sim.nobs)

            plt.plot(sim.x[:sim.nobs,0], sim.f[:sim.nobs],c='k')

    plt.tight_layout()
    plt.savefig("results/simulations/sim-%d.pdf"%k, bbox_inches='tight')

for k in intervals.keys():
    intervals[k] = 1.*intervals[k]/len(sim.datasets)

pd.DataFrame(accuracy).to_csv("results/sim-accuracy.csv",index=False)
