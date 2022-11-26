import argparse
import os
import pickle
import time
from calendar import c
from copy import copy
from os import listdir
from typing import Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import image
from tqdm import tqdm

from calc_entropy import calcEntropyRGB2d

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='all', choices=['acre', 'awa2', 'fish', 'fruits', 'imagenet', 'lego', 'places', 'raven', '2dgeometric'],
                    help='figure the dataset')
parser.add_argument('--path', default='YOUR_PATH_TO_DATA',
                    help='path to dataset (default: ./data)')
parser.add_argument('--savepath', default='./ent',
                    help='path to save results (default: ./ent)')

def analyze_dataset(concepts: Union[set, list], images: dict):
    '''
    Calculating concept dataset.

    @Args:

        concepts: all concepts in the dataset;

        images: images with corresponding concepts as keys;

    @Return:

        list(dict(key: concept, value: tuple(mean, var)))
    '''
    return [
        # (mean, var)
        __analyze_concept(images[concept], concept)
        for concept
        in tqdm(concepts, desc="outer", position=0)
    ]


def __analyze_concept(images_given_concept: Union[set, list], concept: str):
    '''
    Analyze image entropy within a concept.

    @Args:

        images_given_concept: S(image|concept)

    @Return:

        dict(key: concept, value: tuple(mean, var));
    '''
    ents = np.array(
        [
            calcEntropyRGB2d(img, (224, 224))
            for img 
            in tqdm(images_given_concept, desc="inner", position=1, leave=False)
        ]
    )
    
    return {concept: (ents.mean(), ents.var())}

def main():
    args = parser.parse_args()
    concepts = []
    images = {}
    if args.dataset == 'awa2':
        path = args.path + '/' + args.dataset + '/Animals_with_Attributes2/JPEGImages'
        conceptList = listdir(path)
        for concept in tqdm(conceptList, desc="concept", position=0):
            concepts.append(concept)
            imglist = os.listdir(path+f'/{concept}')
            # imglist.sort()
            imgs = []
            for i in tqdm(imglist, desc="pic", position=1, leave=False):
                img = cv2.imread(os.path.join(path+f'/{concept}', i))
                imgs.append(img)
            images[concept] = imgs.copy()
    elif args.dataset == 'fruits':
        path = args.path + '/' + args.dataset + '/Training'
        conceptList = listdir(path)
        for concept in tqdm(conceptList, desc="concept", position=0):
            concepts.append(concept)
            imglist = os.listdir(path+f'/{concept}')
            # imglist.sort()
            imgs = []
            for i in tqdm(imglist, desc="pic", position=1, leave=False):
                img = cv2.imread(os.path.join(path+f'/{concept}', i))
                imgs.append(img)
            images[concept] = imgs.copy()
    elif args.dataset == 'lego':
        path = args.path + '/' + args.dataset + '/lego_v1'
        conceptList = listdir(path)
        for concept in tqdm(conceptList, desc="concept", position=0):
            concepts.append(concept)
            imglist = os.listdir(path+f'/{concept}')
            # imglist.sort()
            imgs = []
            for i in tqdm(imglist, desc="pic", position=1, leave=False):
                img = cv2.imread(os.path.join(path+f'/{concept}', i))
                imgs.append(img)
            images[concept] = imgs.copy()
    elif args.dataset == 'imagenet':
        path = args.path + '/' + args.dataset + '/train'
        conceptList = listdir(path)
        map_dict = {}
        with open(args.path + '/' + args.dataset + '/LOC_synset_mapping.txt', 'r') as f:
            line = f.readline().split(" ", 1)
            while line:
                if len(line) < 2:
                    break
                map_dict[line[0]] = line[1]
                line = f.readline().split(" ", 1)
        for concept in tqdm(conceptList, desc="concept", position=0):
            concept_des = map_dict[concept]
            concepts.append(concept_des)
            imglist = os.listdir(path+f'/{concept}')
            # imglist.sort()
            imgs = []
            for i in tqdm(imglist, desc="pic", position=1, leave=False):
                img = cv2.imread(os.path.join(path+f'/{concept}', i))
                imgs.append(img)
            images[concept_des] = imgs.copy()
    elif args.dataset == 'fish':
        path = args.path + '/' + args.dataset + '/WildFish_part1'
        conceptList = []
        imglist = os.listdir(path)
        # imglist.sort()
        for i in tqdm(imglist, desc="pic", position=0, leave=False):
            img = cv2.imread(os.path.join(path, i))
            if img is None:
                continue
            concept = "_".join(i.split("_", 2)[:2])
            if concept not in concepts:
                concepts.append(concept)
                images[concept] = []
            images[concept].append(img.copy())
    elif args.dataset == 'raven':
        path = args.path + '/' + args.dataset
        conceptList = listdir(path)
        for concept in tqdm(conceptList, desc="concept", position=0):
            concepts.append(concept)
            imglist = os.listdir(path+f'/{concept}')
            # imglist.sort()
            imgs = []
            for i in tqdm(imglist, desc="pic", position=1, leave=False):
                img = cv2.imread(os.path.join(path+f'/{concept}', i))
                imgs.append(img)
            images[concept] = imgs.copy()
    elif args.dataset == 'acre':
        path = args.path + '/' + args.dataset
        conceptList = listdir(path)
        for concept in tqdm(conceptList, desc="concept", position=0):
            concepts.append(concept)
            imglist = os.listdir(path+f'/{concept}')
            # imglist.sort()
            imgs = []
            for i in tqdm(imglist, desc="pic", position=1, leave=False):
                img = cv2.imread(os.path.join(path+f'/{concept}', i))
                imgs.append(img)
            images[concept] = imgs.copy()
    elif args.dataset == '2dgeometric':
        path = args.path + '/' + args.dataset
        conceptList = listdir(path)
        for concept in tqdm(conceptList, desc="concept", position=0):
            concepts.append(concept)
            imglist = os.listdir(path+f'/{concept}')
            # imglist.sort()
            imgs = []
            for i in tqdm(imglist, desc="pic", position=1, leave=False):
                img = cv2.imread(os.path.join(path+f'/{concept}', i))
                imgs.append(img)
            images[concept] = imgs.copy()
    elif args.dataset == 'places.bak':
        path = args.path + '/' + args.dataset + '/val_256'
        imglist = os.listdir(path)
        map_dict = {}
        pic_dict = {}
        with open(args.path + '/' + args.dataset + '/categories_places365.txt', 'r') as f:
            line = f.readline().split(" ", 1)
            while line:
                if len(line) < 2:
                    break
                if line[1][-1] == '\n':
                    line[1] = line[1][:-1]
                map_dict[line[1]] = line[0]
                line = f.readline().split(" ", 1)
        conceptList = listdir(path)
        with open(args.path + '/' + args.dataset + '/places365_val.txt', 'r') as f:
            line = f.readline().split(" ", 1)
            while line:
                if len(line) < 2:
                    break
                if line[1][-1] == '\n':
                    line[1] = line[1][:-1]
                pic_dict[line[0]] = line[1]
                line = f.readline().split(" ", 1)
        conceptList = listdir(path)
        for pic in tqdm(imglist):
            img = cv2.imread(os.path.join(path, pic))
            concept = map_dict[pic_dict[pic]]
            if concept not in concepts:
                concepts.append(concept)
                images[concept] = []
            images[concept].append(img.copy())
    res = analyze_dataset(concepts=concepts, images=images)

    if not os.path.isdir(args.savepath):
        os.mkdir(args.savepath)
    
    with open(args.savepath + '/' + args.dataset + '.pk', 'wb') as ent:
        pickle.dump(res, ent)
    ent.close()


if __name__ == '__main__':
    main()


