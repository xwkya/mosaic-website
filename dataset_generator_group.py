import multiprocessing as mp
from os import name
import pickle
import shelve
import cv2
from strategies import AverageStrategyCosineFaiss
from image_generator import get_image, img_to_tiles
import os
import csv
import requests # to get image from the web
import shutil # to save it locally
import time
import numpy as np

def pprint(*args):
    print(mp.current_process().name, "|", *args)

def generate_label(tile_list, strategy, n_div = 10): 
    '''
        return the index of k_best. Because the strategy uses name_to_index, they are correctly indexed.
    '''
    start = time.time()
    k_best, distances = strategy.find_best(tile_list, get_dist=True) #[(name, average_tile)]
    print(len(tile_list), round(time.time()-start,2))
    return k_best, distances

class ParrMemory:
    def __init__(self, genq=None, resq=None):
        '''
            n_tiles: number of tiles to be generated horizontally and vertically
        '''
        self.genq = genq
        self.resq = resq
    
    def add_new(self, strategy):
        url, tile_list = self.genq.get()
        tile_list_len = len(tile_list)
        tiles_shape = tile_list[0].shape

        tile_list = np.array(list(map(self.group_transform, tile_list))) # tile_list = [[tile1+rotations], ..., [tilen+rotations]]
        tile_list = tile_list.reshape(8*tile_list_len, *tiles_shape) # [tiles]

        labels, distances = generate_label(tile_list, strategy) # [labels], [distances]
        labels = labels.reshape(-1, 8)
        distances = distances.reshape(-1, 8)
        tile_list = tile_list.reshape(tile_list_len, 8, *tiles_shape)
       
        for i in range(distances.shape[0]):
            self.resq.put((url, tile_list[i, 0], labels[i, :], distances[i, :]) )
            
    
    @staticmethod
    def group_transform(tile):
        new_list = []
        
        new_list.append(tile)
        new_list.append(cv2.rotate(tile, cv2.ROTATE_90_CLOCKWISE))
        new_list.append(cv2.rotate(tile, cv2.ROTATE_180))
        new_list.append(cv2.rotate(tile, cv2.ROTATE_90_COUNTERCLOCKWISE))
        
        tile = cv2.flip(tile, 0)

        new_list.append(tile)
        new_list.append(cv2.rotate(tile, cv2.ROTATE_90_CLOCKWISE))
        new_list.append(cv2.rotate(tile, cv2.ROTATE_180))
        new_list.append(cv2.rotate(tile, cv2.ROTATE_90_COUNTERCLOCKWISE))
        
        return new_list
        
def gen_tiles(genq, n_tiles):
    '''
        Puts tiles in a genq list
    '''
    def get_url(reader):
        return next(reader)[0]

    reader = csv.reader(open("train_images.tsv"), delimiter="\t")
    next(reader)

    while True:
        url = get_url(reader)
        try:
            img = get_image(url)
        except Exception as e:
            print(f"Error in get url: {e}")
            continue
        
        if img is False:
            continue
        
        try:
            tile_list = img_to_tiles(img, n_tiles)
        except Exception as E:
            print("Error in dividing image:",E)

        genq.put((url, tile_list))

def write_queue(resq, file_name):
    i=0
    with open(file_name, 'wb') as f:
        while True:
            url, tile, labels, distances = resq.get()
            print(labels, distances)
            pickle.dump((tile, labels, distances), f, -1)
            i+=1
            if (i+1)%500==0:
                print(i)

def generate_n(resq, genq, n, strategy, name_to_index, use_gpu):
    pprint("Generating strategy object..")
    strategy_obj = strategy(name_to_index, limit=None, use_cells=True, scaling=0.8)
    pprint("Done generating strategy object.")
    memory = ParrMemory(genq, resq)
    for i in range(n):
        memory.add_new(strategy_obj)

        

if __name__ == "__main__":
    num_proc = 1

    strategy = AverageStrategyCosineFaiss
    num_gen_per_proc = 10000

    with open("name_to_index.pkl", "rb") as f:
        name_to_index = pickle.load(f)
    
    m = mp.Manager()
    resq = m.Queue(100)
    genq = m.Queue(16)

    gen_process = mp.Process(target=gen_tiles, args=(genq, 32), name="Generator")
    write_process = mp.Process(target=write_queue, args=(resq, "train_data_augmented/dataset.pkl"), name="Writer")

    procs = []
    for i in range(num_proc):
        procs.append(mp.Process(target=generate_n, args=(resq, genq, num_gen_per_proc, strategy, name_to_index, False), name=f"Tiling {i+1}"))

    print("Starting Processes..")
    gen_process.start()

    for proc in procs:
        proc.start()

    write_process.start()

    print("Finished starting processes..")


    gen_process.join()