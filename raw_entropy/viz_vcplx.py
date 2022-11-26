import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


def viz_gens():
    sns.set(style='whitegrid', color_codes=True)
    sns.set_palette('pastel')

    fig = plt.figure()

    # change over diversity
    d = [
        ['LEGO', 1.01],
        ['LEGO', 1.03],
        ['LEGO', 1.083],
        ['LEGO', 1.183],
        ['LEGO', 1.283],
        ['2D-Geo', 1.288],
        ['2D-Geo', 2.018],
        ['2D-Geo', 2.288],
        ['2D-Geo', 2.288],
        ['2D-Geo', 2.423],
        ['ACRE', 4.385],
        ['ACRE', 4.385],
        ['ACRE', 4.390],
        ['ACRE', 4.390],
        ['ACRE', 4.390],
        ['AwA', 5.011],
        ['AwA', 5.521],
        ['AwA', 5.521],
        ['AwA', 5.521],
        ['AwA', 5.521],
        ['Places365', 5.032],
        ['Places365', 5.189],
        ['Places365', 5.579],
        ['Places365', 5.897],
        ['Places365', 6.053],
        ['ImageNet', 4.547],
        ['ImageNet', 5.858],
        ['ImageNet', 6.032],
        ['ImageNet', 6.075],
        ['ImageNet', 6.877],
    ]

    df = pd.DataFrame(d, columns=['dataset', 'visual complexity'])

    sns.set(style='whitegrid', color_codes=True)
    sns.set_palette('pastel')

    sns.set(font_scale = 1.2)
    plt.figure(figsize=(4, 4))
    ax = sns.barplot(
        x = 'dataset', 
        y = 'visual complexity', 
        # hue = 'generalization type', 
        data = df,
        capsize = 0.1
    )
    ax.set_xticklabels(['L.', 'G.', 'A.', 'W.', 'P.', 'I.'])
    ax.xaxis.set_label_text('Datasets')
    ax.yaxis.set_label_text('Log Visual Complexity')

    # ax.legend(loc='upper left')

    plt.savefig('viz/vcplx.pdf', transparent=False, dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()



if __name__ == '__main__':
    viz_gens()

