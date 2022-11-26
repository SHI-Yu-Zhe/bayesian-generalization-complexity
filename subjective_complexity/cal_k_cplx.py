import os
import sys
from unicodedata import category

import torch
import torch.nn as nn

sys.path.append(".")
sys.path.append("..")
import argparse
import csv
import time
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torchvision.models as models
from dataloader import get_dataset
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm, trange

from k_cplx_AR import K_Complexity_AR


loaders = {}
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', metavar='DIR', default='YOUR_PATH_TO_DATA',
                    help='path to datasets (default: YOUR_PATH_TO_DATA)')
parser.add_argument('--dataset', default='all', choices=['2dgeometric', 'acre', 'awa2', 'fish', 'fruits', 'imagenet', 'lego', 'places', 'raven'],
                    help='figure the dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--savepath', default='./ent',
                    help='path to save results (default: ./ent)')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-r', '--resume', default='model_best.pth.tar', type=str,
                    help='path to resume model')
parser.add_argument('-t', '--threshold', default=0, type=float,
                    help='threshold')

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
    
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()

    embedding_dim = 2048
    if args.arch == "resnet18":
        embedding_dim = 512

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
    
    with torch.no_grad():   
        print(f"Get total {len(list(imgdict.keys()))} categories.")
        print(imgdict.keys())
        AR = pd.read_csv(f"./matrices/{args.dataset}_matrix_AR.csv")
        AR = torch.Tensor(AR.iloc[:, 2].to_numpy().reshape(-1, embedding_dim))
        kar = K_Complexity_AR(AR)
        Ks = {}
        thres = float(args.threshold/100)
        print(thres)
        for target in tqdm(sorted(list(imgdict.keys()))):
            images = torch.stack(imgdict[target],dim=1).squeeze(0)
            label = int(train_set.class_to_idx[target])
            # images = imgdict[target][0]
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            # compute output
            embedding = resnet(images)
            kar.init_concept(embedding, label)
            K = 1
            
            for K in range(1, embedding_dim):
                res = fc(avgpool(embedding * (kar.get_mask_K(K).cuda()))[:,:,0,0])
                _, pred = res.topk(1, 1, True, True)
                pred = pred.t().tolist()[0]
                acc = len([i for i in pred if i == label])/len(imgdict[target])
                if acc >= thres:
                    print('Complexity for concept {} is {}.'.format(target, K))
                    Ks[target] = K
                    break
                if K == 2047:
                    print('Complexity for concept {} FAIL.'.format(target))
                    Ks[target] = K
                    break
            
            print(Ks)
            with open(f'k_comp_{args.dataset}.csv', 'w') as f:
                for key in Ks.keys():
                    f.write("%s, %s\n" % (key, Ks[key]))


def run_all_Ks(dataset):
    args = parser.parse_args()  # imagenet
    args.dataset = dataset
    args.resume = f"ckpt/model_best_resnet18_{dataset}.pth.tar"
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()

    embedding_dim = 2048
    if args.arch == "resnet18":
        embedding_dim = 512

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
    
    with torch.no_grad():   
        print(f"Get total {len(list(imgdict.keys()))} categories.")
        print(imgdict.keys())
        AR = pd.read_csv(f"./matrices_{args.arch}/{args.dataset}_matrix_AR.csv")
        AR = torch.Tensor(AR.iloc[:, 2].to_numpy().reshape(-1, embedding_dim))
        kar = K_Complexity_AR(AR)

        # initiate save dict
        K_diffs = dict()
        K_diff_rates = dict()
        K_adj = dict()
        K_acc = dict()

        
        thres = float(args.threshold/100)
        print(thres)
        acc_ctgy = {}
        for target in tqdm(sorted(list(imgdict.keys()))):
            images = torch.stack(imgdict[target],dim=1).squeeze(0)
            label = int(train_set.class_to_idx[target])
            # images = imgdict[target][0]
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            # compute output
            embedding = resnet(images)
            kar.init_concept(embedding, label)
            K = 1

            # First get the baseline accuracy of the concept
            original_res = fc(avgpool(embedding)[:, :, 0, 0])
            _, pred = original_res.topk(1, 1, True, True)
            pred = pred.t().tolist()[0]
            # get acc in current K
            original_acc = len([i for i in pred if i == label])/len(imgdict[target])
            K_acc[target] = original_acc

        
            # save as list
            acc_diff = []
            acc_diff_rate = []
            acc_adj_iter = []


            
            # for K in range(1, embedding_dim):
            #     res = fc(avgpool(embedding * (kar.get_mask_K(K).cuda()))[:,:,0,0])
            #     _, pred = res.topk(1, 1, True, True)
            #     pred = pred.t().tolist()[0]
            #     # get acc in current K
            #     acc = len([i for i in pred if i == label])/len(imgdict[target])
            #     diff = original_acc - acc
                
            #     # append (attribute_id, diff) pair to the list
            #     acc_diff.append((kar.curr_decorder[K-1], diff))
            #     # append (attribute_id, diff_rate) pair to the list
            #     if original_acc == 0:
            #         original_acc = 1e-6
            #     acc_diff_rate.append((kar.curr_decorder[K-1], diff / original_acc))

            #     if K >= 2:
            #         # append (last_attribute_id, next_attribute_id, adj_acc_diff)
            #         acc_adj_iter.append((kar.curr_decorder[K-2], kar.curr_decorder[K-1], last_acc - acc))

            #     # update last_iter
            #     last_acc = acc

            # K_diffs[label] = acc_diff
            # K_diff_rates[label] = acc_diff_rate
            # K_adj[label] = acc_adj_iter

        # # construct datadict with column id as concept name, and each row in each column is a single {attribute_id: diff(or diff_rate)} dictionary
        # df_K_diffs = pd.DataFrame(K_diffs)
        # df_K_diff_rates = pd.DataFrame(K_diff_rates)
        # df_K_adj = pd.DataFrame(K_adj)

        # df_K_diffs.to_csv(f'k_diffs_{args.dataset}.csv', index=False)
        # df_K_diff_rates.to_csv(f'k_diff_rates_{args.dataset}.csv', index=False)
        # df_K_adj.to_csv(f'k_adj_{args.dataset}.csv', index=False)
    df_K_acc = pd.DataFrame(K_acc, index = [0])
    df_K_acc.to_csv(f'k_acc_{args.dataset}.csv')
    return

    
if __name__ == '__main__':
    for cat in ['2dgeometric', 'acre', 'awa2', 'fish', 'fruits', 'imagenet', 'lego', 'places']:
        run_all_Ks(cat)
    # main()
