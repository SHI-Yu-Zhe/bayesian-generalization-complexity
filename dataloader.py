import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

DATASETS = ['imagenet', 'acre', 'awa2', 'raven', 'places.bak', 'lego', 'fruits', '2dgeometric']

class Fish_Dataset(Dataset):
    def __init__(self, datadir, transform=None):
        self.imgdir = datadir + '/part1'
        self.imglist = os.listdir(self.imgdir)
        self.concepts = []
        self.class_to_idx = {}
        idx = 0
        for img in self.imglist:
            concept = "_".join(img.split("_", 2)[:2])
            if concept not in list(self.class_to_idx.keys()):
                self.class_to_idx[concept] = idx
                idx += 1
        self.transform = transform
        
    def __len__(self):
        return len(self.imglist)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = Image.open(os.path.join(self.imgdir, self.imglist[idx])).convert("RGB")
        concept = self.class_to_idx["_".join(self.imglist[idx].split("_", 2)[:2])]
        if self.transform:
            img = self.transform(img)
        sample = (img, int(concept))
        return sample

def load_dataset(path, dataset):

    # normalize and transform
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
    if dataset == 'imagenet':    
        traindir = os.path.join(path, 'train')
        valdir = os.path.join(path, 'val')
        
        train_dataset = datasets.ImageFolder(
            traindir,
            transform)
        val_dataset = datasets.ImageFolder(
            valdir, 
            transform)
        return train_dataset, val_dataset

    # acre/est
    elif dataset == 'acre':
        dataset = datasets.ImageFolder(path,transform)
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len(dataset)-int(len(dataset)/5), int(len(dataset)/5)])
        train_dataset.class_to_idx = dataset.class_to_idx
        val_dataset.class_to_idx = dataset.class_to_idx
        return train_dataset, val_dataset
    
    # awa2
    elif dataset == 'awa2':
        path = path
        dataset = datasets.ImageFolder(path,transform)
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len(dataset)-int(len(dataset)/5), int(len(dataset)/5)])
        train_dataset.class_to_idx = dataset.class_to_idx
        val_dataset.class_to_idx = dataset.class_to_idx
        return train_dataset, val_dataset
    
    # raven
    elif dataset == 'raven':
        dataset = datasets.ImageFolder(path,transform)
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len(dataset)-int(len(dataset)/5), int(len(dataset)/5)])
        train_dataset.class_to_idx = dataset.class_to_idx
        val_dataset.class_to_idx = dataset.class_to_idx
        return train_dataset, val_dataset

    # places
    elif dataset == 'places':
        dataset = Fish_Dataset(datadir=path, transform=transform)
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len(dataset)-int(len(dataset)/5), int(len(dataset)/5)])
        train_dataset.class_to_idx = dataset.class_to_idx
        val_dataset.class_to_idx = dataset.class_to_idx
        return train_dataset, val_dataset

    # elif dataset == 'places.bak':
    #      traindir = os.path.join(path, 'train')
    #      valdir = os.path.join(path, 'val')
    #      train_dataset = datasets.ImageFolder(
    #                         traindir,
    #                     transform)
    #      val_dataset = datasets.ImageFolder(
    #                         valdir,
    #                     transform)
    #      return train_dataset, val_dataset

    # # fish
    # elif dataset == 'fish':
    #     dataset = Fish_Dataset(datadir=path, transform=transform)
    #     train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len(dataset)-int(len(dataset)/5), int(len(dataset)/5)])
    #     train_dataset.class_to_idx = dataset.class_to_idx
    #     val_dataset.class_to_idx = dataset.class_to_idx
    #     return train_dataset, val_dataset
    
    # 2dgeometric
    elif dataset == '2dgeometric':
        dataset = datasets.ImageFolder(
            path,
            transform)
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len(dataset)-int(len(dataset)/5), int(len(dataset)/5)])
        train_dataset.class_to_idx = dataset.class_to_idx
        val_dataset.class_to_idx = dataset.class_to_idx
        return train_dataset, val_dataset
    
    # lego
    elif dataset == 'lego':
        dataset = datasets.ImageFolder(
            path,
            transform)
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len(dataset)-int(len(dataset)/5), int(len(dataset)/5)])
        train_dataset.class_to_idx = dataset.class_to_idx
        val_dataset.class_to_idx = dataset.class_to_idx
        return train_dataset, val_dataset

    # fruits
    elif dataset == 'fruits':
        path = path + '/Training'
        dataset = datasets.ImageFolder(
            path,
            transform)
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len(dataset)-int(len(dataset)/5), int(len(dataset)/5)])
        train_dataset.class_to_idx = dataset.class_to_idx
        val_dataset.class_to_idx = dataset.class_to_idx
        return train_dataset, val_dataset

    # ood
    elif dataset == 'ood':
        traindir = os.path.join(path, 'train')
        valdir = os.path.join(path, 'val')
        train_dataset = datasets.ImageFolder(
            traindir,
            transform)
        val_dataset = datasets.ImageFolder(
            valdir, 
            transform)
        return train_dataset, val_dataset
    
    # ood
    elif dataset == 'all_new':
        traindir = os.path.join(path, 'train')
        valdir = os.path.join(path, 'val')
        train_dataset = datasets.ImageFolder(
            traindir,
            transform)
        val_dataset = datasets.ImageFolder(
            valdir, 
            transform)
        return train_dataset, val_dataset
        
def get_dataset(args):
    train_list = []
    val_list = []
    if args.dataset == 'all':
        for dataset in DATASETS:
            path = os.path.join(args.data, dataset)
            train_set, val_set = load_dataset(path, dataset)
            train_list.append(train_set)
            val_list.append(val_set)
            print(f"Load dataset {dataset} {len(train_set)+len(val_set)} samples")
        train_datasets = torch.utils.data.ConcatDataset(train_list)
        val_datasets = torch.utils.data.ConcatDataset(val_list)
        return train_datasets, val_datasets
    else:
        path = os.path.join(args.data, args.dataset)
        train_set, val_set = load_dataset(path, args.dataset)
        print(f"Load dataset {args.dataset} {len(train_set)+len(val_set)} samples")
        return train_set, val_set


def dataloader(args):
    train_sampler = None
    val_sampler = None
    train_dataset, val_dataset = get_dataset(args)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    return train_loader, val_loader
