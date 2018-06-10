"""Set up and make an MCMC run."""

from emcee.utils import MPIPool
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import subprocess as sp
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import emcee
import plotting
import time

sns.set_style('ticks')


class MCMCrun:
    """Basically just an object that can be plotted I think."""

    def __init__(self, name, nwalkers, path=None, burn_in=0):

        # read in chain from .csv
        if path:
            self.main = pd.read_csv(path + '.csv')
        else:
            pd.read_csv(name + '/' + name + '_chain.csv')
        self.name = name
        self.nwalkers = nwalkers
        self.nsteps = self.main.shape[0] // nwalkers

        # Remove burn in
        self.burnt_in = self.main.iloc[burn_in*nwalkers:, :]
        # Get rid of steps that resulted in bad lnprobs
        lnprob_vals = self.burnt_in.loc[:, 'lnprob']
        self.groomed = self.burnt_in.loc[lnprob_vals != -np.inf, :]
        print 'Removed burn-in phase (step 0 through {} ).'.format(burn_in)

    def evolution(self):
        """Plot walker evolution."""
        print 'Making walker evolution plot...'
        plt.close()

        stepmin, stepmax = 0, self.nsteps

        main = self.main.copy().iloc[stepmin * self.nwalkers:
                                     stepmax * self.nwalkers, :]

        axes = main.iloc[0::self.nwalkers].plot(
            x=np.arange(stepmin, stepmax),
            figsize=(7, 2.0*(len(main.columns))),
            subplots=True,
            color='black',
            alpha=0.1)

        for i in range(self.nwalkers-1):
            main.iloc[i+1::self.nwalkers].plot(
                x=np.arange(stepmin, stepmax), subplots=True, ax=axes,
                legend=False, color='black', alpha=0.1)

            # make y-limits on lnprob subplot reasonable
            axes[-1].set_ylim(main.iloc[-1 * self.nwalkers:, -
                                        1].min(), main.lnprob.max())

        # if you want mean at each step over plotted:
        # main.index //= self.nwalkers
        # walker_means = pd.DataFrame([main.loc[i].mean() for i in range(self.nsteps)])
        # walker_means.plot(subplots=True, ax=axes, legend=False, color='forestgreen', ls='--')

        plt.suptitle(self.name + ' walker evolution')
        plt.savefig(self.name + '/' + self.name +
                    '_evolution.png'.format(self.name))  # , dpi=1)

    def kde(self):
        """Make a kernel density estimate (KDE) plot."""
        print 'Generating posterior kde plots...'
        plt.close()

        nrows, ncols = (2, int(np.ceil((self.groomed.shape[1] - 1) / 2.)))
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.5*ncols, 2.5*nrows))

        # plot kde of each free parameter
        for i, param in enumerate(self.groomed.columns[:-1]):
            ax = axes.flatten()[i]

            for tick in ax.get_xticklabels():
                tick.set_rotation(30)
            ax.set_title(param)
            ax.tick_params(axis='y', left='off', labelleft='off')

            samples = self.groomed.loc[:, param]
            plotting.my_kde(samples, ax=ax)

            percentiles = samples.quantile([.16, .5, .84])
            ax.axvline(percentiles.iloc[0], lw=1,
                       ls='dotted', color='k', alpha=0.5)
            ax.axvline(percentiles.iloc[1], lw=1.5, ls='dotted', color='k')
            ax.axvline(percentiles.iloc[2], lw=1,
                       ls='dotted', color='k', alpha=0.5)

        # hide unfilled axes
        for ax in axes.flatten()[self.groomed.shape[1]:]:
            ax.set_axis_off()

        # bivariate kde to fill last subplot
        # ax = axes.flatten()[-1]
        # for tick in ax.get_xticklabels(): tick.set_rotation(30)
        # sns.kdeplot(self.groomed[r'$i$ ($\degree$)'], self.groomed[r'Scale Factor'], shade=True, cmap='Blues', n_levels=6, ax=ax);
        # ax.tick_params(axis='y', left='off', labelleft='off', right='on', labelright='on')

        # adjust spacing and save
        plt.tight_layout()
        plt.savefig(self.name + '/' + self.name + '_kde.png'.format(self.name))
        plt.show()

    def corner(self, variables=None):
        """Plot 'corner plot' of fit."""
        plt.close()

        # get best_fit and posterior statistics
        stats = self.groomed.describe(percentiles=[0.16, 0.84]).drop([
            'count', 'min', 'max', 'mean'])
        stats.loc['best fit'] = self.main.loc[self.main['lnprob'].idxmax()]
        stats = stats.iloc[[-1]].append(stats.iloc[:-1])
        stats.loc[['16%', '84%'], :] -= stats.loc['50%', :]
        stats = stats.reindex(
            ['50%', '16%', '84%', 'best fit', 'std'], copy=False)
        print(stats.T.round(6).to_string())

        # make corner plot
        corner = sns.PairGrid(data=self.groomed, diag_sharey=False, despine=False,
                              vars=variables)

        if variables is not None:
            corner.map_lower(plt.scatter, s=1, color='#708090', alpha=0.1)
        else:
            corner.map_lower(sns.kdeplot, cut=0, cmap='Blues',
                             n_levels=18, shade=True)

        corner.map_lower(sns.kdeplot, cut=0, cmap='Blues',
                         n_levels=3, shade=False)
        corner.map_diag(sns.kdeplot, cut=0)

        if variables is None:
            # get best_fit and posterior statistics
            stats = self.groomed.describe(percentiles=[0.16, 0.84]).drop([
                'count', 'min', 'max', 'mean'])
            stats.loc['best fit'] = self.main.loc[self.main['lnprob'].idxmax()]
            stats = stats.iloc[[-1]].append(stats.iloc[:-1])
            stats.loc[['16%', '84%'], :] -= stats.loc['50%', :]
            stats = stats.reindex(
                ['50%', '16%', '84%', 'best fit', 'std'], copy=False)
            print(stats.T.round(3).to_string())
            # print(stats.round(2).to_latex())

            # add stats to corner plot as table
            table_ax = corner.fig.add_axes([0, 0, 1, 1], frameon=False)
            table_ax.axis('off')
            left, bottom = 0.15, 0.83
            pd.plotting.table(table_ax, stats.round(2), bbox=[
                              left, bottom, 1-left, .12], edges='open', colLoc='right')

            corner.fig.suptitle(r'{} Parameters, {} Walkers, {} Steps $\to$ {} Samples'
                                .format(self.groomed.shape[1], self.nwalkers,
                                        self.groomed.shape[0]//self.nwalkers, self.groomed.shape[0],
                                        fontsize=25))
            tag = ''
        else:
            tag = '_subset'

        # hide upper triangle, so that it's a conventional corner plot
        for i, j in zip(*np.triu_indices_from(corner.axes, 1)):
            corner.axes[i, j].set_visible(False)

        # fix decimal representation
        for ax in corner.axes.flat:
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.3g'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.3g'))

        plt.subplots_adjust(top=0.9)
        plt.savefig(self.name + '/' + self.name + tag +
                    '_corner.png'.format(self.name))


def run_emcee(run_name, nsteps, nwalkers, lnprob, to_vary):
    """The heart of it.

    Args:
        run_name (str): the name to output I guess
        nsteps (int):
        nwalkers (int):
        lnprob (something):
        to_vary (list of lists): list of [param name,
                                          initial_position_center,
                                          initial_position_sigma,
                                          (prior low bound, prior high bound)]
                                for each parameter.
                                The second two values set the position & size
                                for a random Gaussian ball of initial positions
    """
    # Something to do with parallelizing
    pool = MPIPool()
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    # initiate sampler chain
    ndim = len(to_vary)
    sampler = emcee.EnsembleSampler(nwalkers,
                                    ndim,
                                    lnprob,
                                    args=(run_name, to_vary),
                                    pool=pool)

    # Name the chain we're looking for
    chain_filename = run_name + '/' + run_name + '_chain.csv'
    # Try to resume an existing run of this name.
    try:
        # Read in an existing chain
        chain = pd.read_csv(chain_filename)
        start_step = chain.index[-1] // nwalkers
        print 'Resuming {} at step {}'.format(run_name, start_step)
        pos = np.array(chain.iloc[-nwalkers:, :-1])

        # If we're adding new steps, just put in a new line and get started.
        with open(chain_filename, 'a') as f:
            f.write('\n')
        end = np.array(chain.iloc[-nwalkers:, :])
        print 'Start step: {}'.format(np.mean(end[:, -1]))

    # If there's no pre-existing run, set one up.
    except IOError:

        sp.call(['mkdir', run_name])
        sp.call(['mkdir', run_name + '/model_files'])

        print 'Starting {}'.format(run_name)

        start_step = 0
        # Start a new file for the chain
        # Set up a header line
        with open(chain_filename, 'w') as f:
            param_names = [param[0] for param in to_vary]
            np.savetxt(f,
                       (np.append(param_names, 'lnprob'), ),
                       delimiter=',', fmt='%s')

        # Set up initial positions?
        """I think this is saying the same thing as the nested list comps.
        pos = []
        for i in range(nwalkers):
            for param in to_vary:
                pos.append(param[1] + param[2]*np.random.randn())
                """
        # randn makes an n-dimensional array of rands in [0,1]
        pos = [[param[1] + param[2]*np.random.randn() for param in to_vary]
               for i in range(nwalkers)]

    # Initialize the lnprob list
    lnprobs = []
    first_sample = sampler.sample(pos, iterations=nsteps, storechain=False)

    for i, result in enumerate(first_sample):
        """Enumerate returns a tuple the element and a counter.
            tuples = [t for t in enumerate(['a', 'b', 'c'])]
            counters = [c for c, l in enumerate(['a', 'b', 'c'])]
            """
        old_lnprobs = np.copy(lnprobs)
        pos, lnprobs, blob = result
        print "Step {}: {}".format(start_step + i, np.mean(lnprobs))
        # print('Acceptances: {}'.format([lnprob for lnprob in lnprobs if lnprob not in old_lnprobs]))
        # print('')
        # print(lnprobs)
        # print(np.mean(pos))

        # Log out the new positions
        with open(chain_filename, 'a') as f:
            new_step = [np.append(pos[k], lnprobs[k]) for k in range(nwalkers)]
            np.savetxt(f, new_step, delimiter=',')

    pool.close()


def run_emcee_simple(run_name, nsteps, nwalkers, lnprob, to_vary, burn_in=0, pool=False, resume=False):
    """A new version of run_emcee.

    Args:
        run_name (str):
        nsteps (int):
        nwalkers (int):
        lnprob (function?): I think lnprob here is a function?
                            Maybe equivalent in docs to lnpostfn
        to_vary (list of lists):
        burn_in (int?): how many steps to remove from the front
        pool (bool): Want to parallelize?
        resume (bool): Are you resuming a previous run?
    """
    # Set up parallelization
    if pool:
        pool = MPIPool()
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

    start = time.time()
    # initiate sampler chain
    ndim = len(to_vary[0])
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)

    # Name the chain we're looking for
    chain_filename = run_name + '/' + run_name + '_chain.csv'

    # This seems a little risky bc if you forget, you just overwrite.
    if resume:
        chain = pd.read_csv(chain_filename)
        start_step = chain.index[-1] // nwalkers
        print('Resuming {} at step {}'.format(run_name, start_step))

        with open(chain_filename, 'a') as f:
            f.write('\n')
        pos = np.array(chain.iloc[-nwalkers:, :-1])

    else:
        sp.call('rm -rf ' + run_name, shell=True)
        sp.call(['mkdir', run_name])
        print('Starting {}'.format(run_name))
        start_step = 0

        with open(chain_filename, 'w') as f:
            f.write(','.join([param[0]
                              for param in to_vary] + ['lnprob']) + '\n')
        pos = [[param[1] + param[2]*np.random.randn() for param in to_vary]
               for i in range(nwalkers)]

    # Run the sampler and then query it
    run = sampler.sample(pos, iterations=nsteps, storechain=False)
    """Note that sampler.sample returns:
            pos: list of the walkers' current positions in an object of shape
                    [nwalkers, ndim]
            lnprob: The list of log posterior probabilities for the walkers at
                    positions given by pos.
                    The shape of this object is (nwalkers, dim)
            rstate: The current state of the random number generator.
            blobs (optional): The metadata "blobs" associated with the current
                              position. The value is only returned if
                              lnpostfn returns blobs too.
            """
    for i, result in enumerate(run):
        print "Step {}".format(start_step + i)
        # Where did chisum come from? What is it returning?
        pos, chisum, blob = result
        with open(run_name + '/' + run_name + '_chain.csv', 'a') as f:
            for i in range(nwalkers):
                f.write(
                    ','.join(map(str, np.append(pos[i], chisum[i]))) + '\n')
    print('{} samples in {:.1f} seconds'.format(
        nsteps*nwalkers, time.time() - start))

    if pool:
        pool.close()

    return MCMCrun(run_name, nwalkers, burn_in=burn_in)
