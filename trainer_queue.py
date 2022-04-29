from fileinput import filename
from socket import timeout
import threading
import queue
import cv2
import pickle
import numpy as np
import torch
from nn_models import NNConstructed, NNpolicy_torchresize
from image_generator import get_image, img_to_tiles
import multiprocessing as mp
import os
import ctypes
import torchvision

DOWNLOAD_BUFFER_SIZE = 40

def pprint(*args):
    print(mp.current_process().name, "|", *args)

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


def read_dataset(read_q, filename):
    with open(filename, "rb") as conn:
        while True:
            try:
                url, labels = pickle.load(conn) #labels have shape (576, 10)
                read_q.put((url,labels))
            except EOFError:
                conn.seek(0)
    

def data_produce(q, read_q, n_tiles ):
    while True:
        url, labels = read_q.get()
        
        img = get_image(url, str=mp.current_process().name)
        try:
            tile_list = img_to_tiles(img, n_tiles=n_tiles)
            q.put((tile_list, labels))
        except:
            continue

def train(q, NN_class, NN_args, max_iter, n_tiles, load=False, total_loss_history=None):
    NN = NN_class(*NN_args)
    if load:
        NN.load()
        NN.train()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    NN.to(device)
    loss_fun = torch.nn.CrossEntropyLoss()
    decay = 1.01
    lr = 3e-4
    optimizer = torch.optim.Adam(NN.parameters(), lr=lr)
    init_n = 0
    n_samples = init_n
    loss_history = []
    resizer = torchvision.transforms.Resize((32, 32))

    while n_samples < max_iter:
        # Decay LR
        if n_samples%25 == 0:
            lr /= decay
            for g in optimizer.param_groups:
                g['lr'] = lr

        if n_samples%1000 == 0 and n_samples != 0:
            pprint(f"Loss iteration {n_samples}: {np.mean(loss_history)}")
            loss_history = []

        if n_samples%100 == 0 and n_samples != 0:
            #NN.save(n_samples)
            NN.save()

        tile_list, labels = q.get()
        
        optimizer.zero_grad()
            
        image_batch = torch.Tensor(np.array(tile_list)).transpose(1,3)
        image_batch = resizer(image_batch).transpose(1,3)
        image_batch = image_batch.to(device)/255
        image_batch.requires_grad = True
        labels_batch = torch.LongTensor( labels ).to(device) # (bsz * n_tiles,)
        # Train the network
        preds = NN(image_batch)
        loss = loss_fun(preds, labels_batch)
        loss_history.append(loss.item())
        total_loss_history.append(loss.item())
        loss.backward()
        #print(self.NN.conv1.weight.grad.max(), self.NN.linear1.weight.grad.max(), preds.grad.max()
        optimizer.step()

        
        n_samples +=1
    

def create_generator_processes(num_generators, q, read_q, n_tiles):
    procs = []
    for i in range(num_generators):
        procs.append(mp.Process(target=data_produce, args=(q, read_q, n_tiles), name=f"Generator {i+1}"))
    
    return procs

def start_training(NN_class, NN_args, num_generators, generate_n):
    n_tiles = 32 # Dataset specific value
    filename = "dataset.pkl" # Name of the dataset file

    num_training_cycles = generate_n*num_generators
    manager = mp.Manager()
    q = manager.Queue(DOWNLOAD_BUFFER_SIZE)
    read_q = manager.Queue(4*num_generators)
    hist = manager.list([])

    dataset_reader = mp.Process(target=read_dataset, args=(read_q, filename), name="Dataset Reader")
    procs = create_generator_processes(num_generators, q, read_q, n_tiles)
    trainer = mp.Process(target=train, args=(q, NN_class, NN_args, num_training_cycles, n_tiles, False, hist), name="Trainer")
    dataset_reader.start()
    for proc in procs:
        proc.start()
    trainer.start()

    trainer.join()
    for proc in procs:
        proc.join(timeout=1.0)
    dataset_reader.join(timeout=1.0)

    return hist

if __name__ == '__main__':
    with open("name_to_index.pkl", 'rb') as f:
        name_to_index = pickle.load(f)

    NN_class = NNConstructed
    NN_args = (10000, name_to_index, "cosine", True, True)
    print(str(NN_args[3:]))
    num_generators = 4
    

    # Define the processes
    

    loss = start_training(NN_class, NN_args, num_generators, 1000)
    