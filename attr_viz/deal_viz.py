
import os
import sys
from unicodedata import category
import torch
import torch.nn as nn
import cv2
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
from viz_attr import viz_attr
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
from subjective_complexity.k_cplx_AR import K_Complexity_AR
from PIL import Image



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

def GetKey(val, dict):
    for key, value in dict.items():
        if val == value:
            return key
    return "key doesn't exist"

def main(dataset, model, values):
    ctgys = []
    units = {}
    for value in values:
        ctgys.append(value[0])
        units[value[0]] = value[1]
    args = parser.parse_args()  # imagenet
    args.dataset = dataset
    if model == '152':
        args.arch = 'resnet152'
        args.resume = f'ckpt/model_best_{args.arch}_{dataset}.pth.tar'
    elif model == '101':
        args.arch = 'resnet101'
        args.resume = f'ckpt/model_best_{args.arch}_{dataset}.pth.tar'
    elif model == '18':
        args.arch = 'resnet18'
        args.resume = f'ckpt/model_best_{args.arch}_{dataset}.pth.tar'
    
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
                if len(imgdict[label]) < 1:
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

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

        transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        
        for n, target in enumerate(sorted(list(imgdict.keys()))):
            if n not in ctgys:
                continue
            attrs = kar.AR[n, :]
            curr_decorder = torch.argsort(attrs, dim=0, descending=True).tolist()[:20+1]
            datasetpath = os.path.join(args.data, args.dataset)
            if args.dataset == 'lego':
                datasetpath = os.path.join(datasetpath, "lego_v1", target)
            elif args.dataset == 'imagenet':
                datasetpath = os.path.join(datasetpath, "val", target)
            elif args.dataset == 'awa2':
                datasetpath = os.path.join(datasetpath, "Animals_with_Attributes2/JPEGImages", target) 
            elif args.dataset == 'places':
                datasetpath = os.path.join(datasetpath, "val", target)        
            else:
                datasetpath = os.path.join(datasetpath, target)
            for i in range(5):
                img_name = os.listdir(datasetpath)[i]
                img = Image.open(os.path.join(datasetpath, img_name)).convert("RGB")
                img_trans = transform(img)
                images = torch.stack([img_trans],dim=0)
                label = int(train_set.class_to_idx[target])
                if args.gpu is not None:
                    images = images.cuda(args.gpu, non_blocking=True)
                # compute output
                for e in curr_decorder:
                    embedding = resnet(images)[0, e, :, :]
                    res, mask, realimg = viz_attr(np.array(img_trans), embedding.cpu())
                    im = Image.fromarray(np.uint8(res))
                    if not os.path.isdir(f"./pic/{args.dataset}_{target}_{i}"):
                        os.mkdir(f"./pic/{args.dataset}_{target}_{i}")
                    Image.fromarray(np.uint8(realimg)).save(f"./pic/{args.dataset}_{target}_{i}/realimg.jpg")
                    Image.fromarray(np.uint8(mask)).save(f"./pic/{args.dataset}_{target}_{i}/mask_{e}.jpg")
                    im.save(f"./pic/{args.dataset}_{target}_{i}/res_{e}.jpg")


def get_dict():
    dir_path = "YOUR_CASE_STUDY_PATH"
    with open(dir_path, 'r+') as f:
        lines  = f.readlines()

    ex_dict = {}
    for line in lines:
        if line[0] == '(':
            key = line.split(':')[0].replace('\', \'', '_')[2:-2]
        elif '0' <= line[0] <= '9':
            values = []
            values.append(int(line.split(':')[0])) 
            values.append(int(line.split(':')[1][:-2]))
            if key not in list(ex_dict.keys()):
                ex_dict[key] = []
            ex_dict[key].append(values)
    return(ex_dict)
    
if __name__ == '__main__':
    exs = get_dict()
    for key in list(exs.keys()):
        dataset = key.split('_')[0]
        model = key.split('_')[1]
        main(dataset, model, exs[key])


