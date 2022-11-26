import argparse
import math
import os
import pickle

import numpy as np

from vis_all_entropy import VisEnt
from vis_regression import vis_regplot, vis_barboxplot

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='./ent',
                    help='path to dataset (default:./ent)')
parser.add_argument('--dataset', default='all', choices=['acre','sel_acre', 'awa2', 'fish', 'fruits', 'imagenet', 'lego', 'places', 'raven', 'imagenet_artifacts', 'imagenet_animals', 'sel_all'],
                    help='figure the dataset')
parser.add_argument('--save', default=False,
                    help='path to save results (default: ./ent)')

def load_ent(args, path):
    objects = []
    with open(path, "rb") as openfile:
        while True:
            try:
                objects += pickle.load(openfile)
            except EOFError:
                break

    try:
        # running over 1 specified dataset
        return {args.dataset: objects}
    except:
        # running over all datasets
        if 'sel' in args:
            return {args[4:]: objects}
        else:
            return {args: objects}


if __name__ == '__main__':
    args = parser.parse_args()
    # dataset = {
    #     'imagenet': [
    #         {'cat', (4, 4.5)},
    #         {'dog', (5, 3.5)},
    #     ],
    #     'place': [
    #         {'cafe', (3, 1)},
    #         {'shore', (2, 2)}
    #     ],
    # }
    
    if args.dataset in "all":
        # run all datasets
        dataset_names = [  
            'lego', 
            '2dgeometric',
            'acre',
            'awa2', 
            'places', 
            'imagenet',
        ]

        dataset = dict()
        for dn in dataset_names:
            dataset.update(load_ent(dn, os.path.join(args.path, dn) + '.pk'))
    elif args.dataset in "sel_all":
        # run all datasets
        dataset_names = [  
            # 'fish', 
            'lego', 
            # '2dgeometric'
            'fruits', 
            'acre',
            # 'raven',
            'imagenet_artifacts',
            'awa2', 
            'places', 
            'imagenet_animals',
        ]

        dataset_names = [
            'sel_{}'.format(dname)
            for dname
            in dataset_names
        ]

        dataset = dict()
        for dn in dataset_names:
            dataset.update(load_ent(dn, os.path.join(args.path, dn) + '.pk'))
    else:       
        dataset = load_ent(args, os.path.join(args.path, args.dataset) + '.pk')
    # print(dataset)
    # exit()

    for d in dataset:
        dset = dataset[d]
        mean = 0.
        var = 0.
        for cdict in dset:
            mean += list(cdict.values())[0][0]
            var += list(cdict.values())[0][1]
        
        meanavg = mean / len(dset)
        logmeanavg = np.log(mean / np.sqrt(len(dset)))
        # print(meanavg, logmeanavg)
    
    # exit()
    
    vis = VisEnt()
    
    # vis.vis_all_entropy_pixel(dataset)
    # vis.vis_all_entropy_RT(dataset, args.save)

    # vis_regplot(dataset)

    vis_barboxplot(dataset)

