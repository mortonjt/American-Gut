import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cmx
from matplotlib.ticker import LogFormatter
import pandas as pd
from biom import load_table
import scipy.stats as sps
import statsmodels.api as sm



def plot_scatter(fig, ax,
                 samp_df,
                 metavar,
                 alpha=0.75,
                 zorder_dict=None,
                 sample_cmap=None,
                 sample_legend=1):
    """
    fig : matplotlib.fig
    ax : matplotlib.Axes

    """
    i=0
    if len(set(samp_df[metavar])) < 10:
        for name, group in samp_df.groupby(metavar):
            if zorder_dict != None:
                z = zorder_dict[name]
            else:
                z=i
            if sample_cmap != None:
                col = sample_cmap[name]
                l1 = ax.plot(group["PCA1"], group["PCA2"],
                             marker='o', linestyle='', ms=8,c=col,
                             label=name, alpha=alpha, zorder=z)
            else:
                ax.plot(group["PCA1"], group["PCA2"],
                        marker='o', linestyle='', ms=8,
                        label=name, alpha=alpha, zorder=z)
        i+=1
        ax.legend(title=metavar, loc=sample_legend)

    else:
        cNorm = matplotlib.colors.Normalize(vmin=min(samp_df[metavar]), vmax=max(samp_df[metavar]))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap='seismic')
        ax.scatter(samp_df["PCA1"], samp_df["PCA2"],
                   c=scalarMap.to_rgba(samp_df[metavar]), s=50,
                   alpha=alpha, lw=0,
                   zorder=2)
        scalarMap.set_array(samp_df[metavar])
        lvls = np.linspace(min(samp_df[metavar]), max(samp_df[metavar]),10)
        fig.colorbar(scalarMap,label=metavar, ticks=lvls)

    return fig, ax

def plot_contour(fig, ax,
                 samp_df,
                 metavar,
                 alpha=0.75,
                 zorder_dict=None,
                 sample_cmap=None,
                 sample_legend=1):
    lines, labels = [], []
    m1, m2 = samp_df["PCA1"], samp_df["PCA2"]
    xmin = m1.min()
    xmax = m1.max()
    ymin = m2.min()
    ymax = m2.max()


    K = len(samp_df.groupby(metavar).groups.keys())
    M = max(zorder_dict.values())
    alpha_dict = {}
    for zor in zorder_dict.keys():
        alpha_dict[zor] = alpha * ((M-zorder_dict[zor]+1)/float(K+1))**2

    for name, group in samp_df.groupby(metavar):
        m1, m2 = group["PCA1"], group["PCA2"]
        m1 = m1.fillna(0)
        m2 = m2.fillna(0)
        X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        if zorder_dict != None:
            z = zorder_dict[name]
        else:
            z=i
        assert sample_cmap != None

        try:
            col = sample_cmap[name]
            values = np.vstack([m1, m2])
            kernel = sps.gaussian_kde(values)
            Z = np.reshape(kernel(positions).T, X.shape)
            #ax.imshow(np.rot90(Z), cmap=col,
            #          extent=[xmin, xmax, ymin, ymax])
            norm = matplotlib.colors.Normalize(vmax=Z.max(), vmin=2*Z.min())
            ax.contourf(X, Y, Z, cmap=col, norm=norm, alpha=alpha_dict[name], zorder=z)
            lines.append(mpatches.Rectangle((0,0),1,1,fc=col(50)))
            labels.append(name)
        except:
            continue

    ax.legend(lines, labels, title=metavar, loc=sample_legend)
    return fig, ax


def make_biplot(samp_df,
                feat_df,
                metavar,
                eigvals,
                otu_var="phylum",
                spread="scatter",
                alpha=0.75,
                zorder_dict=None,
                sample_cmap=None,
                otu_cmap=None,
                sample_legend=1,
                otu_legend=2):

    if otu_cmap == None:
        otu_cmap={'p__Bacteroidetes':'#219F8D',
              'p__Firmicutes':'#6CAD3F',
              'p__Proteobacteria':'#D4D71C',
              'p__Verrucomicrobia':'#DFAC35',
              'p__Derferribateres':'#CF5635',
              'p__Cyanobacteria':'#CD4050',
              'p__Tenericutes':'#D04984',
              'p__Actinobacteria':'#7C4A87',
              'Other':'#1394CA'}

    fig, ax = plt.subplots(figsize=(12,12))
    feat_df = feat_df.sort(ascending=False)
    pca1 = np.ravel(samp_df['PCA1'])
    pca2 = np.ravel(samp_df['PCA2'])
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    i=0
    recs=[]
    phyla = []
    num_features = len(feat_df.index)
    for name, group in feat_df.groupby(otu_var):
        #name = otu.split(';')[1].replace(' ','')
        if name in otu_cmap:
            c = otu_cmap[name]
            phylum = name
        else:
            c = otu_cmap['Other']
            phylum = 'Other'
        for otu in group.index:
            a = np.asscalar(feat_df.loc[otu]['PCA1'])
            b = np.asscalar(feat_df.loc[otu]['PCA2'])
            r = np.asscalar(feat_df.loc[otu]['radius'])
            theta = np.asscalar(feat_df.loc[otu]['degrees'])

            plt.arrow(0, 0, a, b, color=c,
                      width=0.02, head_width=0.05, zorder=num_features)
            #plt.text(a, b, genus, color='k', ha='center', va='center')
        if phylum not in phyla:
            recs.append(mpatches.Rectangle((0,0),1,1,fc=c))
            phyla.append(phylum)
    l1 = ax.legend(recs,phyla,loc=otu_legend)

    if spread=="scatter":
        fig, ax = plot_scatter(fig, ax,
                                   samp_df,
                                   metavar,
                                   alpha=alpha,
                                   zorder_dict=zorder_dict,
                                   sample_cmap=sample_cmap,
                                   sample_legend=sample_legend)
    elif spread=="contour":
        fig, ax = plot_contour(fig, ax,
                                   samp_df,
                                   metavar,
                                   alpha=alpha,
                                   zorder_dict=zorder_dict,
                                   sample_cmap=sample_cmap,
                                   sample_legend=sample_legend)

    if len(set(samp_df[metavar])) < 10:
        plt.gca().add_artist(l1)
    # Create some padding
    xmin = min([min(samp_df['PCA1']), min(feat_df['PCA1'])])
    xmax = max([max(samp_df['PCA1']), max(feat_df['PCA1'])])
    ymin = min([min(samp_df['PCA2']), min(feat_df['PCA2'])])
    ymax = max([max(samp_df['PCA2']), max(feat_df['PCA2'])])
    xpad = (xmax - xmin) * 0.1
    ypad = (ymax - ymin) * 0.1
    plt.xlim(xmin - xpad, xmax + xpad)
    plt.ylim(ymin - ypad, ymax + ypad)
    plt.xlabel('PC 1 ({:.2%})'.format(eigvals[0]**2/sum(eigvals**2)))
    plt.ylabel('PC 2 ({:.2%})'.format(eigvals[1]**2/sum(eigvals**2)))
    return fig
