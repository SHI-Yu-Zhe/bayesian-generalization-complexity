from unicodedata import category
import torch
import torch.nn as nn
import os
import sys
sys.path.append(".")
sys.path.append("..")
import argparse
import torchvision.models as models
from dataloader import get_dataset
from torch.utils.data import DataLoader, Subset
from enum import Enum
import time
from attribute_representativeness import Attribute_Representativeness
from viz_matrix import viz_matrix
from sklearn.metrics import confusion_matrix
import seaborn
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

loaders = {}
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', metavar='DIR', default='YOUR_PATH_TO_DATA',
                    help='path to datasets (default: YOUR_PATH_TO_DATA)')
parser.add_argument('--dataset', default='all', choices=['2dgeometric', 'acre', 'awa2', 'fish', 'fruits', 'imagenet', 'lego', 'places', 'raven'],
                    help='figure the dataset')
parser.add_argument('--savepath', default='./ent',
                    help='path to save results (default: ./ent)')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-r', '--resume', default='model_best.pth.tar', type=str,
                    help='path to resume model')
parser.add_argument('--viz', default='run_with_viz', choices=['run_only', 'viz_only'], help='visualization mode')

def accuracy(output, target, lines=None, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        # print("pred: ", pred[0][0], output[0][pred[0][0]], lines[pred[0][0]])
        # print("true: ", target[0], output[0][target[0]], lines[target[0]])
        # print("-"*40)
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def GetKey(val, dict):
    for key, value in dict.items():
        if val == value:
            return key
    return "key doesn't exist"

def main():
    args = parser.parse_args()  # imagenet

    if not args.viz in 'run_only':
        viz_matrix(f'./matrices/{args.dataset}_matrix_AR.csv', 'Attribute Representativeness Matrix', args.dataset)
        viz_matrix(f'./matrices/{args.dataset}_matrix_row.csv', 'P(a|c) Row Matrix', args.dataset)
        viz_matrix(f'./matrices/{args.dataset}_matrix_col.csv', 'P(c\'|a) Column Matrix', args.dataset)
        
        if args.viz in 'viz_only':
            exit()
    
    attrep = Attribute_Representativeness()
    
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    modules = list(model.children())[:-2]
    resnet = nn.Sequential(*modules)
    modules_avgpool = list(model.children())[-2]
    avgpool = nn.Sequential(modules_avgpool)
    modules_fc = list(model.children())[-1]
    fc = nn.Sequential(modules_fc)

    if not torch.cuda.is_available():
    # if 1:
        print('using CPU, this will be slow')
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        resnet = resnet.cuda(args.gpu)
        avgpool = avgpool.cuda(args.gpu)
        fc = fc.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        resnet = torch.nn.DataParallel(resnet).cuda()
        fc = torch.nn.DataParallel(fc).cuda()
        avgpool = torch.nn.DataParallel(avgpool).cuda()

    # eval
    resnet.eval()
    avgpool.eval()
    fc.eval()
    results = []
    imgdict = {}

    if args.dataset != 'all':
        train_set, val_set = get_dataset(args)
        if args.dataset == 'places':
            train_set = val_set
        loader = torch.utils.data.DataLoader(
            train_set, batch_size=1, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=None)
        
        for i, (images, target) in enumerate(loader):
            label = GetKey(target, train_set.class_to_idx)
            try:
                if len(imgdict[label]) < 50:
                    imgdict[label].append(images)
            except:
                imgdict[label] = []
                imgdict[label].append(images)
    
    else:
        # DATASETS = ['imagenet', 'acre', 'awa2', 'raven', 'places', 'lego', 'fruits', '2dgeometric']
        DATASETS = ['imagenet', 'acre', 'places', 'lego', '2dgeometric', 'ood']
        for dataset in DATASETS:
            args.dataset = dataset
            train_set, val_set = get_dataset(args)
            if args.dataset == 'places':
                train_set = val_set
            loader = torch.utils.data.DataLoader(
                train_set, batch_size=1, shuffle=False,
                num_workers=args.workers, pin_memory=True, sampler=None)
            for i, (images, target) in enumerate(loader):
                label = GetKey(target, train_set.class_to_idx)
                try:
                    if len(imgdict[label]) < 50:
                        imgdict[label].append(images)
                except:
                    imgdict[label] = []
                    imgdict[label].append(images)
        args.dataset = 'all'
    
    with torch.no_grad():   
        print(f"Get total {len(list(imgdict.keys()))} categories.")
        print(imgdict.keys())
        for target in tqdm(sorted(list(imgdict.keys()))):
            images = torch.stack(imgdict[target],dim=1).squeeze(0)
            # images = imgdict[target][0]
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            # compute output
            embedding = resnet(images)
            res = fc(avgpool(embedding)[:,:,0,0])
            topk=(1, 5)
            _, pred = res.topk(1, 1, True, True)
            results += pred.t().tolist()[0]
            # print(results)
            # print(embedding.shape)
            attrep.matrix_row = embedding.cpu()
        
        # # confusion_matrix
        # con_matrix = confusion_matrix(sorted(labels), results)
        # print(con_matrix)
        # heatmap = seaborn.heatmap(con_matrix)
        # plt.savefig(args.dataset + ".png", dpi=300)  
        
        attrep.calc_matrix_col()
        attrep.calc_matrix_AR()  
        attrep.AR_stat()
        attrep.calc_embedding_complexity()
        attrep.save_csv(sorted(list(imgdict.keys())), args.dataset)
    
if __name__ == '__main__':
    main()
