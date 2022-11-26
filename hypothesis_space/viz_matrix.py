import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


def viz_matrix(dpath:str, vizname:str, dataset:str):
    '''
    Visualize matrix.

    @Input:

        dpath [str]: relative path to the matrix csv file;

        vizname [str]: diagram name;
    '''
    matdf = pd.read_csv(dpath)

    matpt = matdf.pivot_table(
        index = 'Concepts',
        columns = 'Attributes',
        values = 'Values',
    )

    # print(matpt)

    print(vizname, matpt.values.max(), matpt.values.min())

    ax = plt.figure()
    index = np.argsort(matpt.std())
    plt.figure(figsize=(18, 6))
    ax = sns.heatmap(matpt.iloc[:,index[-60:]], cmap='coolwarm')
    ax.set(
        xlabel = 'Concepts',
        ylabel = 'Attributes',
    )

    plt.title(vizname)

    if not os.path.isdir('./viz_matrices/'):
        os.mkdir('./viz_matrices/')
    
    plt.savefig(f'./viz_matrices/{dataset}_{vizname}.jpg')
    return

