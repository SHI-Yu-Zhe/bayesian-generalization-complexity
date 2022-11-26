import argparse
import os
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='./ent',
                    help='path to dataset (default:./ent)')
parser.add_argument('--dataset', default='all', choices=['acre', 'awa2', 'fish', 'fruits', 'imagenet', 'lego', 'places', 'raven', 'imagenet_artifacts', 'imagenet_animals'],
                    help='figure the dataset')


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
        return {args: objects}

def sel_dataset(args, dname:str, dvalue:list):
    print('Selecting dataset {}...'.format(dname))

    dataset_sel = [
        item
        for item
        in dvalue
        if list(item.values())[0][1] < 5.0
    ]
    
    print("{}: Before {} Concepts, After {} Concepts.".format(dname, len(dvalue), len(dataset_sel)))
    with open(os.path.join(args.path, 'sel_{}.pk'.format(dname)), 'wb') as openfile:
        pickle.dump(dataset_sel, openfile)
    openfile.close()
    return


if __name__ == '__main__':
    args = parser.parse_args()
    
    if args.dataset in "all":
        # run all datasets
        dataset_names = [  
            # 'fish', 
            'fruits', 
            'acre',
            'lego', 
            'raven',
            'imagenet_artifacts', 
            'imagenet_animals',
            'places', 
            'awa2', 
        ]

        dataset = dict()
        for dn in dataset_names:
            dataset.update(load_ent(dn, os.path.join(args.path, dn) + '.pk'))
    else:       
        dataset = load_ent(args, os.path.join(args.path, args.dataset) + '.pk')

    for dname, dvalue in dataset.items():
        sel_dataset(args, dname, dvalue)
