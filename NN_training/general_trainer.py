import pickle
import numpy as np
import torch
from mosaic_project.nn_models import NNGCNN2, NNpolicy_torchresize
from mosaic_project.image_generator import get_image, img_to_tiles
import os
import ctypes
import random
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
    def __init__(self, resize_size, data_path):
        self.resizer = torchvision.transforms.Resize((resize_size, resize_size))
        self.conn = open(data_path, 'rb')
        self.data_path = data_path
    
    
    def __iter__(self):
        while True:
            try:
                tile, labels, distances = pickle.load(self.conn)
            except:
                self.conn.close()
                self.conn = open(self.data_path, 'rb')
                tile, labels, distances = pickle.load(self.conn)
                break
            tile = torch.LongTensor(tile).transpose(0,2)
            tile = self.resizer(tile)/255
            tile = tile.transpose(0, 2)
            distances = torch.Tensor(distances)
            # (distances-torch.mean(distances, dim=-1))/torch.pow(torch.pow(distances-torch.mean(distances,dim=-1), 2).mean(dim=-1), 1/2)
            yield tile, torch.LongTensor(labels), distances/100000
            

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
        parser.add_argument("--no_scheduler", dest='scheduler', default=True, action='store_false')
        parser.add_argument("--lr", type=float, default=3e-4)
        parser.add_argument("--decay", type=float, default=1/1.0005)
        parser.add_argument("--predict_loss", dest='predict_loss', default=False, action='store_true')
        return parent_parser

    def __init__(self, NN_name, load=True, batch_size=1024, scheduler=True, lr=3e-4, decay=1/1.0005, predict_loss=False, data_path='../data/dataset_augmented.pkl', path='', **kwargs):
        super().__init__()
        self.path = path

        self.NN_name = NN_name
        with open(self.path+"name_to_index.pkl", 'rb') as f:
            name_to_index = pickle.load(f)

        if NN_name == 'GCNN':
            NN_class = NNGCNN2 # (out_features, out_groups, name_to_index, name, predict_loss=True)
            NN_args = (10000, 8, name_to_index, "cosine_GCNN", predict_loss)
        elif NN_name == 'CNN':
            NN_class = NNpolicy_torchresize # (out_features, name_to_index, name)
            NN_args = (10000, name_to_index, "cosine_CNN")
        else:
            raise Exception('Unknown NN class name.')
        

        self.NN = NN_class(*NN_args)
        self.batch_size = batch_size
        if load:
            self.train_dataset = CustomDataset(8, data_path)
            self.use_scheduler = scheduler
            self.decay = decay
            self.lr = lr
        self.predict_loss = predict_loss

    def forward(self, x):
        return self.NN(x)

    def on_epoch_start(self):
        print(self.lr_schedulers().get_last_lr())

    def training_step(self, batch, batch_nb):
        '''
            Three cases:
             - GCNN and predict loss: 8 labels are predicted and the 8 corresponding mse distance loss
             - GCNN and not predict loss: 8 labels are predicted, no matter the loss for each rotation.
             - CNN: the best label is predicted, so only 1/8th of the data is utilized.
        '''
        
        x, labels, distances = batch
        best_transforms = torch.argmin(distances, dim=-1)
        # (bsz, ..) (bsz, 8), (bsz, 8)
        if self.predict_loss and self.NN_name == 'GCNN':
            pred_label, pred_transform = self(x) 
            # pred_label: (bsz, 8, 10000)  pred_dist: (bsz, 8)
            label_loss = F.cross_entropy(pred_label.view(pred_label.shape[0]*8,-1), labels.view(-1,))
            distance_loss = F.cross_entropy(pred_transform, best_transforms)
            loss = label_loss + distance_loss
            acc = self.accuracy(pred_transform, best_transforms)
            self.log("performance", {"label_loss": label_loss, "transform_loss": distance_loss, "acc": acc}, prog_bar=True)
            return loss
        
        elif self.NN_name == 'GCNN' and not self.predict_loss:
            pred_label = self(x) 
            # pred_label: (bsz, 8, 10000)
            loss = F.cross_entropy(pred_label.view(pred_label.size(0)*8,-1), labels.view(-1,))
            self.log("loss", loss, prog_bar=True)
            return loss

        else:
            labels = labels[range(labels.size(0)), best_transforms]

            rotation = 90*torch.remainder(best_transforms, 4).type_as(x)
            symmetry = best_transforms.div(4, rounding_mode='floor')

            flipped_x = kornia.geometry.vflip(x)
            x = flipped_x * symmetry[:, None, None, None] + x * (1-symmetry)[:, None, None, None]
            x = kornia.geometry.rotate(x, rotation)

            pred = self(x)
            loss = F.cross_entropy(pred, labels)
            self.log('loss', loss, prog_bar=True)
            return loss
    
    def accuracy(self, x, y):
        return (torch.sum(torch.argmin(x, dim=-1) == y)/x.size(0)).item()

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = {
            'scheduler': ExponentialLR(opt, self.decay),
            "interval": "epoch",
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
    
    model = LitModel(**vars(args), path='../', data_path='dataset_augmented.pkl')
    trainer.fit(model)
    
