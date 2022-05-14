import pickle
import numpy as np
import torch
from nn_models import NNGCNN, NNGCNN2, NNpolicy_torchresize
from image_generator import get_image, img_to_tiles
import os
import ctypes
import torchvision
from torch.autograd import Variable
import kornia
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningModule, Trainer
import torch.nn.functional as F
from pytorch_lightning.plugins import DDPPlugin
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR
from argparse import ArgumentParser


class CustomDataset(Dataset):
    def __init__(self, names, resize_size):
        self.names = names
        self.resizer = torchvision.transforms.Resize((resize_size, resize_size))

    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, index):
        try:
            with open('train_data_augmented/'+self.names[index], 'rb') as f:
                tile, labels, distances =  pickle.load(f)
        except EOFError:
            return None, None, None

        tile = torch.LongTensor(tile).transpose(0,2)
        tile = self.resizer(tile)/255
        tile = tile.transpose(0, 2)
        distances = torch.Tensor(distances)
        # (distances-torch.mean(distances, dim=-1))/torch.pow(torch.pow(distances-torch.mean(distances,dim=-1), 2).mean(dim=-1), 1/2)
        return tile, torch.LongTensor(labels), distances/10000

def create_datasets(eps, resize_size):
    names = os.listdir('train_data_augmented/')
    train_names = names[:int(len(names)*eps)]
    test_names = names[int(len(names)*eps):]
    return CustomDataset(train_names, resize_size), CustomDataset(test_names, resize_size)

train_ds, test_ds = create_datasets(0.8, 32)

def collate_batch_fn(batch):
    image_batch = torch.stack([x[0] for x in batch if x[0] is not None])
    labels_batch = torch.stack([x[1] for x in batch if x[0] is not None])
    distances_batch = torch.stack([x[2] for x in batch if x[0] is not None])

    return image_batch, labels_batch, distances_batch

if __name__=="__main__":
    print("T")
    loader = DataLoader(train_ds, 1024, collate_fn=collate_batch_fn, num_workers=8, shuffle=True, persistent_workers=True, prefetch_factor=3)
    it = iter(loader)
    for i in range(100):
        x = next(it)
        print(i)