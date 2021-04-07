
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _kde_color():
    return '#404788FF'


def _diverging_palette():
    return sns.diverging_palette(220, 20, s=90, as_cmap=True)


def plot_connectivity(mat, mean, q, n_sig=None):

    fig, ax = plt.subplots(figsize=(4, 4))
    fig_pos = ax.get_position()
    cbar_ax = fig.add_axes([.905, fig_pos.y0, .02, .3])
    ax = sns.heatmap(mat, vmin=-1, vmax=1, cmap=_diverging_palette(), 
                     square=True, ax=ax, cbar_ax=cbar_ax)
    ax.set_axis_off()

    if n_sig:
        info = (f"Mean r = {mean}\n"
                f"Prop. p<.05 = {n_sig}\n"
                f"Modularity (Q) = {q}")
    else:
        info = (f"Mean r = {mean}\n"
                f"Modularity (Q) = {q}")
    ax.set_title(info)
    plt.show()


def _carpet_plot(tseries, fd, ax):

    x = tseries.values.T
    sd = np.mean(x.std(axis=0))
    vmin = x.mean() - (2 * sd)
    vmax = x.mean() + (2 * sd)

    ax.imshow(x, cmap='gray', vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_yticks([])
    ax.set_yticklabels([])

    # set 5 ticks
    xmax = x.shape[1]
    interval = xmax / 4
    xticks =  np.round(np.arange(0, xmax + interval, interval), 0)
    ax.set_xticks(xticks)

    ax.set_xlabel('Scan')
    ax.set_ylabel('Regions')

    # check if tseries has scans discarded
    diff = fd.shape[0] - tseries.shape[0]
    if diff > 0:
        fd = fd[diff:]

    spikes = np.argwhere(fd > 20).ravel()
    if len(spikes) > 0:
        ax.scatter(x=spikes, y=[ax.get_ylim()[1] - 15] * len(spikes), 
                   marker='v', c='r', s=25, clip_on=False)
    
    for d in ["left", "top", "bottom", "right"]:
        plt.gca().spines[d].set_visible(False)

    return ax
    


def _motion_trace_plot(confounds):
    pass



def plot_scan(tseries, confounds, measures, mat):
    pass



def plot_measures(measures):
    pass

    # mean_fd

    # n_spikes

    # mean_r

    # q

    # mean fd vs mean r 

    # mean fd vs q




def plot_qc_fc(qc_fc):

    fig, ax = plt.subplots(figsize=(3, 2))
    ax = sns.kdeplot(x=qc_fc, color=_kde_color(), alpha=1, linewidth=0, 
                     fill=True, ax=ax)
    ax.set(xlabel='r')
    ax.axvline(0, c='k', lw=1)
    sns.despine()
    
    # show mean
    mean = np.mean(qc_fc)
    x_loc = ax.get_xlim()[1] * .3
    yloc = ax.get_ylim()[1] * .9   
    ax.text(x=x_loc , y=yloc, s=f'mean = {mean:0.3f}')

    fig.subplots_adjust(top=0.9)
    ax.set_title('QC-FC', fontdict={'size': 12})

    fig.show()


def plot_dist_dependence(distances, qc_fc, r):

    g = sns.JointGrid(x=distances, y=qc_fc, ylim=(-1, 1), height=4)
    g.plot_joint(sns.histplot, cmap='viridis', bins=50, cbar=False)
    g.plot_joint(sns.regplot, scatter=False, ci=None, truncate=False, 
                 line_kws=dict(color='m', linewidth=1))
    g.plot_marginals(sns.kdeplot, color=_kde_color(), alpha=1, linewidth=0, 
                     fill=True)
    g.ax_joint.set(xlabel='Euclidean Distance', ylabel='QC-FC', 
                   yticks=np.arange(-1, 1.5, .5))

    # display correlation value
    x_loc = np.max(distances) * .7  
    g.ax_joint.text(x=x_loc , y=.9, s=f'rho = {r:.3f}', )

    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle('Distance Dependence', fontdict={'size': 12})
    plt.show()




