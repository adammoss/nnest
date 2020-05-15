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

    if args.root:
        log_roots = [args.root]
    else:
        log_roots = glob.glob('logs/*')

    if args.dim != 0:
        x_dims = [args.dim]
    else:
        x_dims = range(2, 50)

    for log_root in log_roots:

        print()
        print('------------------------')
        print(log_root)
        print('------------------------')
        print()

        # Find
        log_dim_dirs = [[] for _ in x_dims]
        for ix, x_dim in enumerate(x_dims):
            for log_dir in glob.glob(os.path.join(log_root, 'run*')):
                if os.path.exists(os.path.join(log_dir, 'info', 'params.txt')):
                    with open(os.path.join(log_dir, 'info', 'params.txt')) as f:
                        data = json.load(f)
                    if int(data['x_dim']) == x_dim:
                        log_dim_dirs[ix].append(log_dir)

        for ix, log_dim_dir in enumerate(log_dim_dirs):

            logzs = []
            dlogzs = []
            nlikes = []
            ess = []

            if len(log_dim_dir) > 0:
                print()
                print('--------')
                print('Dim: %s' % x_dims[ix])
                print('--------')
                print()

            for log_dir in log_dim_dir:

                with open(os.path.join(log_dir, 'info', 'params.txt')) as f:
                    data = json.load(f)

                if os.path.exists(os.path.join(log_dir, 'chains', 'chain.txt')):
                    names = ['p%i' % i for i in range(int(data['x_dim']))]
                    labels = [r'x_{%i}' % i for i in range(int(data['x_dim']))]
                    files = getdist.chains.chainFiles(os.path.join(log_dir, 'chains', 'chain.txt'))
                    if data['sampler'] == 'nested':
                        mc = getdist.MCSamples(os.path.join(log_dir, 'chains', 'chain.txt'), names=names, labels=labels,
                                               ignore_rows=0.0, sampler='nested')
                    else:
                        mc = getdist.MCSamples(os.path.join(log_dir, 'chains', 'chain.txt'), names=names, labels=labels,
                                               ignore_rows=0.3)
                    mc.readChains(files)
                    if args.feedback > 0:
                        print(mc.getMargeStats())

                    if args.plot:
                        g = getdist.plots.getSubplotPlotter()
                        g.triangle_plot(mc, filled=True)
                        g.export(os.path.join(os.path.join(log_dir, 'plots', 'triangle.png')))

                if data['sampler'] == 'nested':
                    if os.path.exists(os.path.join(log_dir, 'results', 'final.csv')):
                        results = pd.read_csv(os.path.join(log_dir, 'results', 'final.csv'))
                        print(results)
                        logzs.append(results['logz'])
                        dlogzs.append(results['logzerr'])
                        nlikes.append(results['ncall'])
                        ess.append(np.sum(mc.weights) ** 2 / np.sum(mc.weights ** 2))

            if len(logzs) > 1:
                print()
                print('Num runs: %s' % (len(logzs)))
                print(r'Log Z: $%4.2f \pm %4.2f$' % (np.mean(logzs), np.std(logzs)))
                print(r'Log Z error estimate: $%4.2f \pm %4.2f$' % (np.mean(dlogzs), np.std(dlogzs)))
                print(r'N_like: $%.0f \pm %.0f$' % (np.mean(nlikes), np.std(nlikes)))
                print(r'Posterior ESS: $%.0f \pm %.0f$' % (np.mean(ess), np.std(ess)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--root', type=str, default='')
    parser.add_argument('--dim', type=int, default=0)
    parser.add_argument('--feedback', type=int, default=0)
    parser.add_argument('-plot', action='store_true')

    args = parser.parse_args()
    main(args)
