#!bin/python

import matplotlib
matplotlib.use('Agg')

import argparse, os, popmachine, itertools, time, ConfigParser
from pystan_cache.pystan_cache import caching_stan
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

gp_multi = caching_stan.stan_model(file='stan-models/gp_multi.stan')

machine = popmachine.Machine()

def saveSamples(samples, directory, basedir = '.'):

    if not os.path.exists(basedir):
        os.makedirs(basedir)

    if not os.path.exists(os.path.join(basedir, directory)):
        os.mkdir(os.path.join(basedir, directory))

    for k in samples.keys():
        np.save(open(os.path.join(basedir, directory, k,), 'wb'), samples[k])


parser = argparse.ArgumentParser(description='Run stan sampling model.')

parser.add_argument("--nchains", type=int, default=2, help='number of chains for sampling')
parser.add_argument("--iter", type=int, default=2000, help='number of sampler iterations')
parser.add_argument("--adapt_delta", type=float, default=.8, help='sampler adapt delta')
parser.add_argument('--step', type=int, default=1, help='step size for thinning data')
parser.add_argument('--designs', help='designs used in model')
parser.add_argument('--outputDir', default='.', help='output directory of model')
parser.add_argument("model", help='model file')
parser.add_argument("search", nargs='*', help='search terms')

args = parser.parse_args()

################################################################
# Data setup
################################################################

kwargs = dict([s.split("=") for s in args.search])
for k in kwargs.keys():
    if ',' in kwargs[k]:
        kwargs[k] = kwargs[k].split(',')

ds = machine.search(**kwargs)

x = ds.data.index.values
x = (x-x.min())/x.max()

y = ds.data.values

y = np.log2(y)
y = (y-y.mean())/y.std()

stehp = args.step
x = x[::stehp]
y = y[::stehp, :]


################################################################
# Design setup
################################################################

factors = []

for d in args.designs.split(','):
    factors.append(ds.meta[d].factorize()[0])

dm = np.zeros((y.shape[1], 1+sum([max(f)+1 for f in factors])))
dm[:,0] = 1

ind = 1
for f in factors:
    for j,k in enumerate(f):
        dm[j,k+ind] = 1
    ind += max(f) + 1

################################################################
# Sampler setup
################################################################

train_data = {
    'N': x.shape[0],
    'P':y.shape[1],
    'K':dm.shape[1],
    'L':1+len(factors),
    'prior': list(itertools.chain.from_iterable([[1]] + [[i+2]*(max(f)+1) for i,f in enumerate(factors)])),
    'design': dm,
    'x': x,
    'y': y.T,
}

train_data['alpha_prior'] = [[1,1]] * train_data['L']
train_data['length_scale_prior'] = [[1,1]]  * train_data['L']

################################################################
# Output
################################################################

label = "_".join([
                    os.path.basename(args.model),
                    ','.join([k+"="+str(v) for k,v in kwargs.iteritems()]),
                    args.designs
                ])

_id = str(int(time.time()))

print 'run_stan'
print 'model: %s' % os.path.basename(args.model)
print 'data: %s' % ' '.join([k+"="+str(v) for k,v in kwargs.iteritems()])
print 'designs: %s' % args.designs
print 'id: %s' % _id


################################################################
# Sample
################################################################

tsamples = gp_multi.sampling(data=train_data, chains=args.nchains, iter=args.iter, control = {'adapt_delta': args.adapt_delta})
tsamp = tsamples.extract(permuted=True)

################################################################
# Save
################################################################

saveSamples(tsamp, _id, basedir=args.outputDir)

config = ConfigParser.RawConfigParser()
config.optionxform = str

config.add_section('base')
config.set('base', 'model', args.model)
config.set('base', 'designs', args.designs)

config.add_section('data')
for k,v in kwargs.iteritems():
    config.set('data', k, v)

with open('%s/%s/config.cfg'%(args.outputDir,_id), 'wb') as configfile:
    config.write(configfile)

plt.plot(x,y);
plt.savefig(os.path.join(args.outputDir, _id, 'data.pdf'))
