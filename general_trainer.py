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
from torch.utils.data import DataLoader, Dataset, IterableDataset
from pytorch_lightning import LightningModule, Trainer
import torch.nn.functional as F
from pytorch_lightning.plugins import DDPPlugin
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR
from argparse import ArgumentParser


class CustomDataset(IterableDataset):
    def __init__(self, resize_size):
        self.resizer = torchvision.transforms.Resize((resize_size, resize_size))
        self.conn = open('dataset_augmented.pkl', 'rb')
    
    
    def __iter__(self):
        while True:
            try:
                tile, labels, distances = pickle.load(self.conn)
            except:
                self.conn.close()
                self.conn = open('dataset_augmented.pkl', 'rb')
                break
            tile = torch.LongTensor(tile).transpose(0,2)
            tile = self.resizer(tile)/255
            tile = tile.transpose(0, 2)
            distances = torch.Tensor(distances)
            # (distances-torch.mean(distances, dim=-1))/torch.pow(torch.pow(distances-torch.mean(distances,dim=-1), 2).mean(dim=-1), 1/2)
            yield tile, torch.LongTensor(labels), distances/10000
            

def collate_batch_fn(batch):
    image_batch = torch.stack([x[0] for x in batch if x[0] is not None])
    labels_batch = torch.stack([x[1] for x in batch if x[0] is not None])
    distances_batch = torch.stack([x[2] for x in batch if x[0] is not None])

    return image_batch, labels_batch, distances_batch

class LitModel(LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        parser.add_argument("--NN_name", type=str, default='GCNN')
        parser.add_argument("--batch_size", type=int, default=1024)
        parser.add_argument("--num_workers", type=int, default=32)
        parser.add_argument("--no_persistent_workers", dest='persistent_workers', default=True, action='store_false')
        parser.add_argument("--no_scheduler", dest='scheduler', default=True, action='store_false')
        parser.add_argument("--lr", type=float, default=3e-4)
        parser.add_argument("--decay", type=float, default=1/1.0005)
        parser.add_argument("--predict_group", dest='predict_group', default=False, action='store_true')
        return parent_parser

    def __init__(self, NN_name, batch_size=1024, num_workers=16, scheduler=True, lr=3e-4, decay=1.0005, predict_group=True, persistent_workers=True, **kwargs):
        super().__init__()

        print('PREDICT GROUP', predict_group)
        with open("name_to_index.pkl", 'rb') as f:
            name_to_index = pickle.load(f)

        if NN_name == 'GCNN':
            NN_class = NNGCNN2
            NN_args = (10000, 8, name_to_index, "cosine_GCNN", predict_group)
            resize_size = 8
        elif NN_name == 'CNN':
            NN_class = NNpolicy_torchresize
            NN_args = (10000, name_to_index, "cosine_CNN")
            resize_size = 32
        else:
            raise Exception('Unknown NN class name.')
        
        self.NN = NN_class(*NN_args)
        self.batch_size = batch_size
        self.train_dataset = CustomDataset(8)
        self.use_scheduler = scheduler
        self.decay = decay
        self.lr = lr
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.predict_group = predict_group

    def forward(self, x):
        return self.NN(x)

    def training_step(self, batch, batch_nb):
        x, labels, distances = batch
        # (bsz, ..) (bsz, 8), (bsz, 8)
        if self.predict_group:
            pred_label, pred_dist = self(x) 
            # pred_label: (bsz, 8, 10000)  pred_dist: (bsz, 8, 1)
            label_loss = F.cross_entropy(pred_label.view(pred_label.shape[0]*8,-1), labels.view(-1,))
            distance_loss = F.mse_loss(pred_dist.squeeze(), distances)
            loss = label_loss + distance_loss
            self.log("performance", {"loss": loss, "label_loss": label_loss, "distance_loss": 10*distance_loss}, prog_bar=True)
            return loss
        
        else:
            pred = self(x)
            loss = F.cross_entropy(pred, labels)
            self.log('loss', loss, prog_bar=True)
            return loss
    

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = {
            'scheduler': ExponentialLR(opt, self.decay)
        }
        if self.use_scheduler:
            return [opt], [scheduler]
        else:
            return opt
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=collate_batch_fn, num_workers=0)
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = LitModel.add_model_specific_args(parser)

    args = parser.parse_args()
    
    trainer = Trainer.from_argparse_args(args)
    
    model = LitModel(**vars(args))
    trainer.fit(model)
    
