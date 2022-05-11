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

def create_datasets(eps, resize_size):
    names = os.listdir('train_data/')
    train_names = names[:int(len(names)*eps)]
    test_names = names[int(len(names)*eps):]
    return CustomDataset(train_names, resize_size), CustomDataset(test_names, resize_size)


class CustomDataset(Dataset):
    def __init__(self, names, resize_size):
        self.names = names
        self.resizer = torchvision.transforms.Resize((resize_size, resize_size))

    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, index):
        try:
            with open('train_data/'+self.names[index], 'rb') as f:
                tile, label =  pickle.load(f)
        except EOFError:
            return None, None

        tile = torch.LongTensor(tile).transpose(0,2)
        tile = self.resizer(tile)/255
        tile = tile.transpose(0, 2)
        return tile, label
            

def collate_batch_fn(batch):
    image_batch, labels_batch = torch.stack([x[0] for x in batch if x[0] is not None]), torch.LongTensor([x[1] for x in batch if x[0] is not None])

    group_batch = torch.randint(0, 8, (labels_batch.shape[0],))
    rotations = torch.remainder(group_batch, 4)*90.
    flip = torch.div(group_batch, 4, rounding_mode='trunc').view(-1,1,1,1)

    image_batch_flipped = kornia.geometry.transform.hflip(image_batch)
    image_batch = image_batch_flipped * flip + image_batch * (1-flip)
    image_batch = kornia.geometry.transform.rotate(image_batch, rotations)

    return image_batch, labels_batch, group_batch

def url_to_images(img_gen, url, n_tiles):
    '''
        returns list of tiles resized to (85, 85)
    '''
    image = img_gen.get_image(url)
    if type(image)==type(False):
        return False
    w, l = image.shape[0], image.shape[1]

    tile_size = min(w//n_tiles, l//n_tiles)
    n_tiles_w = w//tile_size
    n_tiles_l = l//tile_size

    tiles_list = []

    for i in range(n_tiles_w):
        for j in range(n_tiles_l):
            tile_img = image[i*tile_size : (i+1)*tile_size, j*tile_size : (j+1)*tile_size, :]
            #tiles_list.append(cv2.resize(tile_img, (85,85))) # Passing this size to the neural network
            tiles_list.append(tile_img)
            
    return tiles_list



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
            NN_class = NNGCNN
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
        self.train_dataset, self.test_dataset = create_datasets(0.8, resize_size=resize_size)
        self.use_scheduler = scheduler
        self.decay = decay
        self.lr = lr
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.predict_group = predict_group

    def forward(self, x):
        return self.NN(x)

    def training_step(self, batch, batch_nb):
        x, y_feat, y_group = batch
        pred = self(x)
        if self.predict_group:
            loss = F.cross_entropy(pred, y_feat*y_group)
            self.log("performance", {"loss": loss.item()}, prog_bar=True)
            return loss
        
        else:
            loss = F.cross_entropy(pred, y_feat)
            self.log('loss', loss, prog_bar=True)
            return loss
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y_feat, y_group = batch
        pred = self(x)
        if self.predict_group:
            loss = F.cross_entropy(pred, y_feat*y_group)
            self.log("val_loss", loss, prog_bar=True)
            return loss
        
        else:
            loss = F.cross_entropy(pred, y_feat)
            self.log('val_loss', loss, prog_bar=True)
            return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = {
            'scheduler': ExponentialLR(opt, self.decay),
            'interval': 'step'  # called after each training step
        }
        if self.use_scheduler:
            return [opt], [scheduler]
        else:
            return opt
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=collate_batch_fn, num_workers=self.num_workers, shuffle=True, persistent_workers=self.persistent_workers)
    
    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=collate_batch_fn, num_workers=self.num_workers, pin_memory=True, persistent_workers=self.persistent_workers)


def generate_train_dataloader():
    train_dataset, test_dataset = create_datasets(0.8)
    return DataLoader(train_dataset, batch_size=128, collate_fn=collate_batch_fn, num_workers=32 if torch.cuda.is_available() else 0, pin_memory=True if torch.cuda.is_available() else False, shuffle=True)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = LitModel.add_model_specific_args(parser)

    args = parser.parse_args()
    
    trainer = Trainer.from_argparse_args(args)
    
    model = LitModel(**vars(args))
    trainer.fit(model)
    
