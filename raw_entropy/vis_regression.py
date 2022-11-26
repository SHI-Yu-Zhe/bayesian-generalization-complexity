import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cv2 import drawFrameAxes


def vis_regplot(data_concept_ents:dict):
    '''
    Visualize the regressor of all datasets.

    @Args:

        data_concept_ents [dict(key: dataset_name, value: list(tuple(concept_name, mean, var)))];
    '''
    sns.set(style='whitegrid', color_codes=True)
    sns.set_palette('pastel')

    fig = plt.figure()
    dpi = fig.get_dpi()

    mean_list = []
    var_list = []
    for dataset_name, concept_ents in data_concept_ents.items():
        mean_list += [
            list(item.values())[0][0]
            for item
            in concept_ents
        ]
        var_list += [
            list(item.values())[0][1]
            for item
            in concept_ents
        ]

    mean_list = np.array(mean_list)
    var_list = np.array(var_list)
    
    ax = sns.regplot(
        x = mean_list, 
        y = var_list, 
        ci = 95,
        scatter = True,
        fit_reg = True,
        color = "g", 
        marker = "x"
    )

    plt.savefig('raw_ent_reg.pdf', transparent=False, dpi=600, bbox_inches='tight', pad_inches=0)
    return


def vis_barboxplot(dataset):
    sns.set(style='whitegrid', color_codes=True)
    sns.set_palette('pastel')

    fig = plt.figure()

    dname = []
    ent = []

    for ds in dataset:
        for c in dataset[ds]:
            em, ev = list(c.values())[0][0], list(c.values())[0][1]
            e = np.log(em)
            dname.append(database_name(ds))
            ent.append(e)

    ddict = {
        'Dataset': dname,
        'Entropy': ent,
    }
    df = pd.DataFrame(ddict)

    if not os.path.isdir('./viz'):
        os.mkdir('./viz')
    
    ax = sns.barplot(
        x = 'Dataset', 
        y = 'Entropy', 
        data = df,
        ci = 95,
    )

    plt.savefig('viz/raw_ent_bar.pdf', transparent=False, dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()

    ax = sns.boxplot(
        x = 'Dataset', 
        y = 'Entropy', 
        data = df,
        showfliers = False,
    )

    plt.savefig('viz/raw_ent_box.pdf', transparent=False, dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()
    return


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

