import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from ar_example_list import AR_EXAMPLE_LIST


parser = argparse.ArgumentParser()
parser.add_argument('--func', choices=['diag', 'stat'], help='select function')


def sel_in_domain_matrix():
    matdf = pd.read_csv('./viz/all_matrix_AR.csv')
    matrowdf = pd.read_csv('./viz/all_matrix_row.csv')
    matcoldf = pd.read_csv('./viz/all_matrix_col.csv')

    clist = []
    for k in AR_EXAMPLE_LIST:
        clist += list(AR_EXAMPLE_LIST[k].keys())

    for cl in clist:
        if cl in matdf['Concepts']:
            print(cl)
    return


def sel_out_of_domain_matrix(
    matpt:pd.pivot_table, 
    matrowpt:pd.pivot_table, 
    matcolpt:pd.pivot_table,
    name:str,
):

    type_dict = {
        'high AR': set(),
        'low AR': set(),
        'high P(z|c)': set(),
        'low P(z|c)': set(),
        'high P(c|z)': set(),
        'low P(c|z)': set(),
        'high mean, low var': set(),
        'high mean, high var': set(),
        'low mean, high var': set(),
        'low mean, low var': set(),
    }
    
    MAX_AR = 2
    MAX_ROW = 1
    MAX_COL = 1
    MAX_MV = 10

    for tr in matpt.iterrows():
        r = np.array(tr[1])

        # sort decreasing
        ridx = np.argsort(r)[:: -1]

        type_dict['high AR'] = type_dict['high AR'].union(set(ridx[: MAX_AR].tolist()))
        type_dict['low AR'] = type_dict['low AR'].union(set(ridx[- MAX_AR:].tolist()))

    for tr in matrowpt.iterrows():
        r = np.array(tr[1])

        # sort decreasing
        ridx = np.argsort(r)[:: -1]

        type_dict['high P(z|c)'] = type_dict['high P(z|c)'].union(set(ridx[: MAX_ROW].tolist()).difference(type_dict['high AR']))
        type_dict['low P(z|c)'] = type_dict['low P(z|c)'].union(set(ridx[- MAX_ROW:].tolist()).difference(type_dict['low AR']))

    for tr in matcolpt.iterrows():
        r = np.array(tr[1])

        # sort decreasing
        ridx = np.argsort(r)[:: -1]

        type_dict['high P(c|z)'] = type_dict['high P(c|z)'].union(set(ridx[: MAX_COL].tolist()).difference(type_dict['high AR']).union(type_dict['high P(z|c)']))
        type_dict['low P(c|z)'] = type_dict['low P(c|z)'].union(set(ridx[- MAX_COL:].tolist()).difference(type_dict['low AR']).union(type_dict['low P(z|c)']))

    meanlist = []
    varlist = []
    iodlist = []
    for c in matpt:
        mean = matpt[c].values.mean()
        var = matpt[c].values.var()
        iod = var / (mean + 1e-3)
        meanlist.append(mean)
        varlist.append(var)
        iodlist.append(iod)

    msortinc = np.argsort(meanlist)
    msortdec = msortinc[:: -1]
    vsortinc = np.argsort(varlist)
    vsortdec = vsortinc[:: -1]
    iodsortinc = np.argsort(iodlist)
    iodsortdec = iodsortinc[:: -1]

    mhvl = []
    mhvh = []
    mlvh = []
    mlvl = []
    for i in range(2048):
        mhvl.append(msortdec.tolist().index(i) + vsortinc.tolist().index(i))
        mhvh.append(msortdec.tolist().index(i) + vsortdec.tolist().index(i))
        mlvh.append(msortinc.tolist().index(i) + vsortdec.tolist().index(i))
        mlvl.append(msortinc.tolist().index(i) + vsortinc.tolist().index(i))

    mhvlsort = np.argsort(mhvl)
    mhvhsort = np.argsort(mhvh)
    mlvhsort = np.argsort(mlvh)
    mlvlsort = np.argsort(mlvl)

    type_dict['high mean, low var'] = set(mhvlsort[: MAX_MV].tolist())
    type_dict['high mean, high var'] = set(mhvhsort[: MAX_MV].tolist())
    type_dict['low mean, high var'] = set(mlvhsort[: MAX_MV].tolist())
    type_dict['low mean, low var'] = set(mlvlsort[: MAX_MV].tolist())

    mtop = type_dict['high AR'].union(type_dict['high P(z|c)'])
    mtop = mtop.union(type_dict['low P(c|z)'])
    # mtop = mtop.union(type_dict['high mean, high var'])
    mtop = mtop.union(type_dict['high mean, low var'])

    mbtm = type_dict['low AR'].union(type_dict['low P(z|c)'])
    mbtm = mbtm.union(type_dict['high P(c|z)'])
    # mbtm = mbtm.union(type_dict['low mean, high var'])
    mbtm = mbtm.union(type_dict['low mean, low var'])

    selmatpttop = matpt.iloc[:, list(mtop)]
    selmatptbtm = matpt.iloc[:, list(mbtm)] 

    selmatptall = matpt.iloc[:, list(mtop) + list(mbtm)]

    selmatptvardec = matpt.iloc[:, list(vsortdec[: 120]) + list(vsortdec[-60:])]
    selmatptmeandec = matpt.iloc[:, list(msortdec[: 120]) + list(msortdec[-60:])]
    selmatptiodinc = matpt.iloc[:, list(iodsortinc[: 120]) + list(iodsortinc[-60:])]

    figshape = selmatptvardec.shape   

    # sns.set(style='whitegrid', color_codes=True)
    # sns.set_palette('pastel')

    ylabels = [
        'lego block',
        'triangle',
        'star',
        'spot',
        'snowfield',
        'sky',
        'road',
        'tennis',
        'banana',
        'cucumber',
        'watermelon',
        'blue cube',
        'blue cylinder',
        'green cylinder',
        'car on the road',
        'airport',
        'dalmatian',
        'samoyed',
        'spotted tabby',
        'angora cat',
        'cat',
        'mountain bike',
        'motor scooter',
        'plane',
        'car',
    ]


    # plot top + btm
    fig = plt.figure()
    plt.figure(figsize=(round(7 * float(figshape[1]) / float(figshape[0])), 7))

    ax = sns.heatmap(selmatptall, cmap="YlGnBu", cbar=False)
    ax.set(
        xlabel = 'Attributes',
        ylabel = 'Concepts',
    )
    ax.set_yticklabels(ylabels)

    plt.savefig('viz/{}_ar_matrix_all.pdf'.format(name), transparent=False, dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()

    # plot var dec
    fig = plt.figure()
    plt.figure(figsize=(round(6. * float(figshape[1]) / float(figshape[0])), 6))
    
    ax = sns.heatmap(selmatptvardec, cmap="YlGnBu", cbar=False)
    ax.set(
        xlabel = 'Attributes',
        ylabel = 'Concepts',
    )
    ax.set_yticklabels(ylabels)

    plt.savefig('viz/{}_ar_matrix_vardec.pdf'.format(name), transparent=False, dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()

    # plot mean dec
    fig = plt.figure()
    plt.figure(figsize=(round(6. * float(figshape[1]) / float(figshape[0])), 6))
    
    ax = sns.heatmap(selmatptmeandec, cmap="YlGnBu", cbar=False)
    ax.set(
        xlabel = 'Attributes',
        ylabel = 'Concepts',
    )
    ax.set_yticklabels(ylabels)

    plt.savefig('viz/{}_ar_matrix_meandec.pdf'.format(name), transparent=False, dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()

    # plot mean dec
    fig = plt.figure()
    plt.figure(figsize=(round(6. * float(figshape[1]) / float(figshape[0])), 6))
    
    ax = sns.heatmap(selmatptiodinc, cmap="YlGnBu", cbar=False)
    ax.set(
        xlabel = 'Attributes',
        ylabel = 'Concepts',
    )
    ax.set_yticklabels(ylabels)

    plt.savefig('viz/{}_ar_matrix_iodinc.pdf'.format(name), transparent=False, dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()
    return


def viz_diag_matrix(
    matpt:pd.pivot_table, 
    matrowpt:pd.pivot_table, 
    matcolpt:pd.pivot_table,
    name:str,
):
    # diag
    ids = []
    for tr in matpt.iterrows():
        r = np.array(tr[1])

        rsortdec = np.argsort(r)[:: -1]

        # sort decreasing
        maxidx = rsortdec[0]
        i = 1
        while maxidx in ids:
            # resolve duplicated id
            maxidx = rsortdec[i]
            i += 1

        ids.append(maxidx)


    # big diag
    bigids = []
    id_dict = {
        id: 0
        for id 
        in range(25)
    }

    for rnd in range(7):

        rid = 0
        for tr in matpt.iterrows():
            r = np.array(tr[1])

            rsortdec = np.argsort(r)[:: -1]

            # sort decreasing
            i = id_dict[rid]
            maxidx = rsortdec[i]
            while maxidx in bigids:
                i += 1
                # resolve duplicated id
                maxidx = rsortdec[i]
            
            # update index
            id_dict[rid] = i

            # save index according to id position
            bigids.append(maxidx)

            rid += 1

    
    meanlist = []
    varlist = []
    for c in matpt:
        meanlist.append(matpt[c].values.mean())
        varlist.append(matpt[c].values.var())

    msortdec = np.argsort(meanlist)[:: -1]
    vsortinc = np.argsort(varlist)
    vsortdec = vsortinc[:: -1]

    selmatpt = matpt.iloc[:, ids]
    selmatptbig = matpt.iloc[:, bigids]
    selmatptmeandec = matpt.iloc[:, list(msortdec[: 25])]
    selmatptvardec = matpt.iloc[:, list(vsortdec[: 25])]

    ylabels = [
        'lego block',
        'triangle',
        'star',
        'spot',
        'snowfield',
        'sky',
        'road',
        'tennis',
        'banana',
        'cucumber',
        'watermelon',
        'blue cube',
        'blue cylinder',
        'green cylinder',
        'car on the road',
        'airport',
        'dalmatian',
        'samoyed',
        'spotted tabby',
        'angora cat',
        'cat',
        'mountain bike',
        'motor scooter',
        'plane',
        'car',
    ]

    fig = plt.figure()
    plt.figure(figsize=(7.2, 7))
    # sns.set(font_scale = 1.2)

    ax = sns.heatmap(selmatpt, cmap="YlGnBu", cbar=False)
    ax.set(
        xlabel = 'Attributes',
        ylabel = 'Concepts',
    )
    ax.set_yticklabels(ylabels)

    plt.savefig('viz/{}_ar_diag_matrix.pdf'.format(name), transparent=False, dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()


    fig = plt.figure()
    plt.figure(figsize=(42, 6))

    ax = sns.heatmap(selmatptbig, cmap="YlGnBu", cbar=False)
    ax.set(
        xlabel = 'Attributes',
        ylabel = 'Concepts',
    )
    ax.set_yticklabels(ylabels)

    plt.savefig('viz/{}_ar_bigdiag_matrix.pdf'.format(name), transparent=False, dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()

    return
    fig = plt.figure()
    plt.figure(figsize=(9, 7))

    ax = sns.heatmap(selmatptmeandec, cmap="YlGnBu")
    ax.set(
        xlabel = 'Attributes',
        ylabel = 'Concepts',
    )
    ax.set_yticklabels(ylabels)

    plt.savefig('viz/{}_ar_diag_matrix_mdec.pdf'.format(name), transparent=False, dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()


    fig = plt.figure()
    plt.figure(figsize=(9, 7))

    ax = sns.heatmap(selmatptvardec, cmap="YlGnBu")
    ax.set(
        xlabel = 'Attributes',
        ylabel = 'Concepts',
    )
    ax.set_yticklabels(ylabels)

    plt.savefig('viz/{}_ar_diag_matrix_vdec.pdf'.format(name), transparent=False, dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()
    return 


if __name__ == '__main__':
    args = parser.parse_args()

    matdf = pd.read_csv('./viz/resnet152/all_matrix_AR.csv')
    matrowdf = pd.read_csv('./viz/resnet152/all_matrix_row.csv')
    matcoldf = pd.read_csv('./viz/resnet152/all_matrix_col.csv')

    outofdomain = [
        '6632 Technic Lever 3M',
        'triangle',
        'star',
        'spot',
        'snowfield',
        'sky',
        'road',
        'tennis',
        'banana',
        'cucumber',
        'watermelon',
        'SmoothCube_v2MyMetalblue',
        'SmoothCylinderMyMetalblue',
        'SmoothCylinderMyMetalgreen',
        'carfield',
        'airport',
        'n02110341_dalmatian',
        'n02111889_samoyed',
        'spotted_tabby_cat',
        'angora_cat',
        'cat',
        'n03792782_mountain_bike',
        'n03791053_motor_scooter',
        'plane',
        'car',
    ]
    
    indomain = []
    for k in AR_EXAMPLE_LIST:
        indomain += list(AR_EXAMPLE_LIST[k].keys())

    matpt = matdf.pivot_table(
        index = 'Concepts',
        columns = 'Attributes',
        values = 'Values',
    )
    matrowpt = matrowdf.pivot_table(
        index = 'Concepts',
        columns = 'Attributes',
        values = 'Values',
    )
    matcolpt = matcoldf.pivot_table(
        index = 'Concepts',
        columns = 'Attributes',
        values = 'Values',
    )

    matptood = matpt.loc[outofdomain, :]
    matrowptood = matrowpt.loc[outofdomain, :]
    matcolptood = matcolpt.loc[outofdomain, :]

    matptid = matpt.loc[indomain, :]
    matrowptid = matrowpt.loc[indomain, :]
    matcolptid = matcolpt.loc[indomain, :]

    matptood = np.log(matptood)
    matptid = np.log(matptid)
    
    if args.func == 'stat':
        sel_out_of_domain_matrix(matptood, matrowptood, matcolptood, 'out_of_domain')
    elif args.func == 'diag':
        viz_diag_matrix(matptood, matrowptood, matcolptood, 'out_of_domain')

    # sel_out_of_domain_matrix(matptid, matrowptid, matcolptid, 'in_domain')
    

