from __future__ import print_function
from __future__ import division

import glob
import os
import argparse
import json

import pandas as pd
import numpy as np
import getdist
import getdist.plots


def main(args):

    root = os.path.join(args.path, args.name, 'run*')

    logzs = []
    dlogzs = []
    nlikes = []

    for fileroot in glob.glob(root):

        if os.path.exists(os.path.join(fileroot, 'info', 'params.txt')):

            with open(os.path.join(fileroot, 'info', 'params.txt')) as f:
                data = json.load(f)

            if args.x_dim != 0 and int(data['x_dim']) != args.x_dim:
                continue

            print(fileroot)

            if args.sampler == 'nested':
                if os.path.exists(os.path.join(fileroot, 'results', 'final.csv')):
                    results = pd.read_csv(os.path.join(fileroot, 'results', 'final.csv'))
                    print(results)
                    logzs.append(results['logz'])
                    dlogzs.append(results['logzerr'])
                    nlikes.append(results['ncall'])

            if os.path.exists(os.path.join(fileroot, 'chains', 'chain.txt')):
                names = ['p%i' % i for i in range(int(data['x_dim']))]
                labels = [r'x_%i' % i for i in range(int(data['x_dim']))]
                files = getdist.chains.chainFiles(os.path.join(fileroot, 'chains', 'chain.txt'))
                if args.sampler == 'nested':
                    mc = getdist.MCSamples(os.path.join(fileroot, 'chains', 'chain.txt'), names=names, labels=labels,
                                           ignore_rows=0.0, sampler='nested')
                else:
                    mc = getdist.MCSamples(os.path.join(fileroot, 'chains', 'chain.txt'), names=names, labels=labels)
                mc.readChains(files)
                print(mc.getMargeStats())

                if args.plot:
                    g = getdist.plots.getSubplotPlotter()
                    g.triangle_plot(mc, filled=True)
                    g.export(os.path.join(os.path.join(fileroot, 'plots', 'triangle.png')))

    if len(logzs) > 1:
        print(r'Log Z: $%4.2f \pm %4.2f$' % (np.mean(logzs), np.std(logzs)))
        print(r'Log Z error estimate: $%4.2f \pm %4.2f$' % (np.mean(dlogzs), np.std(dlogzs)))
        print(r'N_like: $%.0f \pm %.0f$' % (np.mean(nlikes), np.std(nlikes)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, default='logs')
    parser.add_argument('--name', type=str, default='rosenbrock')
    parser.add_argument('--x_dim', type=int, default=0)
    parser.add_argument('-plot', action='store_true')
    parser.add_argument('--sampler', type=str, default='nested')

    args = parser.parse_args()
    main(args)
