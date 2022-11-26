import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


def plot_kcplx():
    sns.set(style='whitegrid', color_codes=True)
    sns.set_palette('pastel')

    fig = plt.figure()
    dpi = fig.get_dpi()
    # fig.set_size_inches(1920.0 / 3 / float(dpi), 1080.0 / 3 / float(dpi))

    kcplx = pd.read_csv('./res/kcplx.csv')

    # box plot
    ax = sns.boxplot(
        data = kcplx, 
        x = 'databases', 
        y = 'k-complexity', 
        showfliers = False, 
    )

    ax.xaxis.set_label_text('Visual Complexity (Databases)')
    ax.yaxis.set_label_text('Log Subjective Complexity')

    plt.savefig('viz/kcplx_box.pdf', transparent=False, dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()

    # violin plot
    ax = sns.violinplot(
        data = kcplx, 
        x = 'databases', 
        y = 'k-complexity',
        linewidth = 1, 
        cut = 0.2, 
        notch = True, 
        scale = 'count'
    )

    ax.xaxis.set_label_text('Visual Complexity (Databases)')
    ax.yaxis.set_label_text('Log Subjective Complexity')

    plt.savefig('viz/kcplx_violin.pdf', transparent=False, dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()
    return


def plot_valrate():
    sns.set(style='whitegrid', color_codes=True)
    sns.set_palette('pastel')

    fig = plt.figure()
    dpi = fig.get_dpi()
    # fig.set_size_inches(1920.0 / 3 / float(dpi), 1080.0 / 3 / float(dpi))

    val_rate = pd.read_csv('./res/val_rate.csv')

    # box plot
    ax = sns.boxplot(
        data = val_rate, 
        x = 'databases', 
        y = 'validation rate', 
        showfliers = False, 
    )

    ax.xaxis.set_label_text('Visual Complexity (Databases)')
    ax.yaxis.set_label_text('Concept Valid Rate')

    plt.savefig('viz/valrate_box.pdf', transparent=False, dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()

    # violin plot
    ax = sns.violinplot(
        data = val_rate, 
        x = 'databases', 
        y = 'validation rate',
        linewidth = 1, 
        cut = 0.2, 
        notch = True, 
        scale = 'count'
    )

    ax.xaxis.set_label_text('Visual Complexity (Databases)')
    ax.yaxis.set_label_text('Concept Valid Rate')

    plt.savefig('viz/valrate_violin.pdf', transparent=False, dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()
    return 

def plot_kcplx_reg():
    sns.set(style='whitegrid', color_codes=True)
    sns.set_palette('pastel')

    fig = plt.figure()
    dpi = fig.get_dpi()
    # fig.set_size_inches(1920.0 / 3 / float(dpi), 1080.0 / 3 / float(dpi))

    kcplx = pd.read_csv('./res/kcplx.csv')
    
    vcplx = []

    # map to visual complexity
    dmap = {
        'LEGO': 1.083,
        '2D-Geo': 2.288,
        'ACRE': 4.390,
        'AwA': 4.624,
        'Places365': 5.579,
        'ImageNet': 6.075,
    }

    cluster = []
    for kd in kcplx['databases']:
        vcplx.append(dmap[kd])
        cluster.append(0)

    kvdict = {
        'v-complexity': vcplx,
        'k-complexity': kcplx['k-complexity'],
        'cluster': cluster,
    }
    
    kvdf = pd.DataFrame(kvdict)

    # savekvdict = {
    #     'x': vcplx,
    #     'y': kcplx['k-complexity'],
    # }

    # savekvdf = pd.DataFrame(savekvdict)
    # savekvdf.to_csv('./res/v_k_cplx.csv', index=False)
    
    # regression plot
    ax = sns.regplot(
        x = 'v-complexity',
        y = 'k-complexity',
        data = kvdf,
        order = 2,
        ci = 95,
        color = '#bfc5e3',
        marker = 'x',
        scatter_kws = {'marker': '+', 'linewidths': 0.5},
        line_kws = {'linewidth': 2, 'color': '#ff7f8f'},
    )
    
    ax.set_xticks(np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]))

    ax.xaxis.set_label_text('Log Visual Complexity')
    ax.yaxis.set_label_text('Log Subjective Complexity')

    plt.savefig('viz/kcplx_lm.pdf', transparent=False, dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()
    return 


if __name__ == '__main__':
    # plot_kcplx()
    # plot_valrate()
    plot_kcplx_reg()
