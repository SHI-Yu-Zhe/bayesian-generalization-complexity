import itertools
import os

import numpy as np
import pandas as pd


def proc(r, reject:float, converge:float):
    '''
    Data processor.

    @Input:

        r [pandas.DataFrame];

        reject [float]: reject error rate;

        converge [float]: converge error rate;

    @Output:

        tuple(k_cplx, val_rate);
    '''
    reject = float(reject) / 2e2
    converge = float(converge) / 2e2

    blacklist = []
    ks = []

    # iterate over concepts
    for rcol in r:
        col = r[rcol]

        k = 1
        cplx = 0.

        # iterate over attributes
        for c in col:
            # filter out failed concepts
            unit, diff = c.split(', ')
            unit = int(unit[8:-1])
            diff = float(diff[:-1])  
            
            if diff < reject:
                blacklist.append(int(rcol))
                break
            
            diff = 0 if diff < converge else diff
            cplx += diff
            
            # update k
            k += 1
        ks.append(cplx)

    datasize = int(rcol) + 1
    valsize = datasize - len(blacklist)
    
    if valsize == 0 or sum(ks) == 0:
        return False
    else:
        val_rate = valsize / datasize
        k_cplx = np.log(sum(ks) / valsize)
        
        return (k_cplx, val_rate)


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


def k_distribution():
    dp = [
        'res/k_res_18/k_diffs_lego.csv',
        'res/k_res_18/k_diffs_2dgeometric.csv',
        # 'res/k_res_101/k_diffs_fruits.csv',
        'res/k_res_101/k_diffs_acre.csv',
        'res/k_res_101/k_diffs_awa2.csv',
        'res/k_res_152/k_diffs_places.csv',
        'res/k_res_152/k_diffs_imagenet.csv',
    ] 

    # reject rate range
    rangereject = range(-120, -40)
    # converge rate range
    rangeconverge = range(4, 40)

    rejectconverge = list(itertools.product(*(rangereject, rangeconverge)))

    datasetcol = []
    kcplxcol = []
    valratecol = []
    
    for fp in dp:
        dname = database_name(fp)
        print(dname)
        r = pd.read_csv(fp)
        
        for rc in rejectconverge:
            kres = proc(
                r = r,
                reject = rc[0],
                converge = rc[1],
            )
            
            if not kres:
                # no valid concept in dataset
                continue 
            else:
                datasetcol.append(dname)
                kcplxcol.append(kres[0])
                valratecol.append(kres[1])

    kcplxdict = {
        'databases': datasetcol,
        'k-complexity': kcplxcol,
    }

    valratedict = {
        'databases': datasetcol,
        'validation rate': valratecol,
    }

    dfkcplx = pd.DataFrame(kcplxdict)
    dfvalrate = pd.DataFrame(valratedict)

    dfkcplx.to_csv('./res/kcplx.csv', index=False)
    dfvalrate.to_csv('./res/val_rate.csv', index=False)
    return

def acc_k_distribution():
    dp = [
        'res/k_res_18/k_diffs_lego.csv',
        'res/k_res_18/k_diffs_2dgeometric.csv',
        # 'res/k_res_101/k_diffs_fruits.csv',
        'res/k_res_101/k_diffs_acre.csv',
        'res/k_res_101/k_diffs_awa2.csv',
        'res/k_res_152/k_diffs_places.csv',
        'res/k_res_152/k_diffs_imagenet.csv',
    ] 
    return


if __name__ == '__main__':
    k_distribution()
