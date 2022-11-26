import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


def database_name(path:str):
    if 'geometric' in path:
        return '2D-Geo'
    elif 'lego' in path:
        return 'LEGO'
    elif 'fruits' in path:
        return 'Fruits360'
    elif 'acre' in path:
        return 'ACRE'
    elif 'awa2' in path:
        return 'AwA'
    elif 'places' in path:
        return 'Places365'
    elif 'imagenet' in path:
        return 'ImageNet'


def plot_k_acc_distribution():
    dp = [
        'res/k_res_18/k_diffs_lego.csv',
        'res/k_res_18/k_diffs_2dgeometric.csv',
        'res/k_res_101/k_diffs_acre.csv',
        'res/k_res_101/k_diffs_awa2.csv',
        'res/k_res_152/k_diffs_places.csv',
        'res/k_res_152/k_diffs_imagenet.csv',
    ]
    rp = [
        'res/k_acc/acc_resnet18/k_acc_lego.csv',
        'res/k_acc/acc_resnet18/k_acc_2dgeometric.csv',
        'res/k_acc/acc_resnet101/k_acc_acre.csv',
        'res/k_acc/acc_resnet101/k_acc_awa2.csv',
        'res/k_acc/acc_resnet152/k_acc_places.csv',
        'res/k_acc/acc_resnet152/k_acc_imagenet.csv',
    ] 

    datasetcol = []
    kratecol = []
    diffcol = []

    datasetcolkcol = []
    kcol = []
    diffcolkcol = []

    dcolsel = []
    kcolsel = []
    diffcolsel = []

    for fp, frp in zip(dp, rp):
        dname = database_name(fp)
        print(dname)
        df = pd.read_csv(fp)
        rf = pd.read_csv(frp)

        for ccol, rcol in zip(df, rf):
            dfccol = df[ccol]
            rfrcol = rf[rcol][0]

            K = 1
            for cval in dfccol:
                # filter out failed concepts
                unit, diff = cval.split(', ')
                unit = int(unit[8:-1])
                diff = float(diff[:-1])

                datasetcol.append(dname)
                k_rate = (K / (len(dfccol) + 1)) * 100
                kratecol.append(k_rate)

                accdiff = rfrcol - diff
                
                curr_acc = 0. if accdiff < 0. else 1. if accdiff > 1. else accdiff
                curr_acc *= 1e2

                diffcol.append(curr_acc)

                if K < 128:
                    datasetcolkcol.append(dname)
                    kcol.append(K)
                    diffcolkcol.append(curr_acc)

                if K < 11:
                    dcolsel.append(dname)
                    kcolsel.append(K)
                    diffcolsel.append(curr_acc)
                    
                K += 1
                
    k_acc_dict = {
        'Dataset': datasetcol,
        'Subjective Complexity': kratecol,
        'Accuracy': diffcol,
    }

    k_dict = {
        'Dataset': datasetcolkcol,
        'Subjective Complexity': kcol,
        'Accuracy': diffcolkcol,
    }

    k_sel = {
        'D.': dcolsel,
        'Desc. Len.': kcolsel,
        'Accuracy (%)': diffcolsel,
    }

    dfkacc = pd.DataFrame(k_acc_dict)
    dfk = pd.DataFrame(k_dict)
    dfksel = pd.DataFrame(k_sel)
    
    sns.set(style='whitegrid', color_codes=True)
    sns.set_palette('pastel')

    fig = plt.figure()

    # plot K rate
    ax = sns.lineplot(
        x = "Subjective Complexity",
        y = "Accuracy",
        data = dfkacc,
        hue = "Dataset",
    )

    ax.xaxis.set_label_text("Relative Discription Length (%)")
    ax.yaxis.set_label_text("Accuracy (%)")

    ax.legend(loc='best')

    plt.savefig('viz/kcplxaccrate.pdf', transparent=False, dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()

    # plot K
    ax = sns.lineplot(
        x = "Subjective Complexity",
        y = "Accuracy",
        data = dfk,
        hue = "Dataset",
    )

    ax.xaxis.set_label_text("Discription Length")
    ax.yaxis.set_label_text("Accuracy (%)")

    ax.legend(loc='best')

    plt.savefig('viz/kcplxacc.pdf', transparent=False, dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()

    # plot first-K
    sns.set(font_scale = 1.75)

    g = sns.FacetGrid(
        data = dfksel,
        col = "D.",
        col_wrap = 6,
        ylim = (0, 100),
    )

    g.map(
        sns.barplot,
        "Desc. Len.",
        "Accuracy (%)",
        order = list(range(1, 11)),
        ci = 95,
    )

    plt.savefig('viz/kacctop.pdf', transparent=False, dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()
    return 


if __name__ == '__main__':
    plot_k_acc_distribution()
