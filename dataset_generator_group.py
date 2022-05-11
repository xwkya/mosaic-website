import multiprocessing as mp
from os import name
import pickle
import cv2
from strategies import AverageStrategyCosineFaiss
from image_generator import get_image, img_to_tiles
import os
import csv
import requests # to get image from the web
import shutil # to save it locally
import time

def pprint(*args):
    print(mp.current_process().name, "|", *args)

def generate_label(tile_list, strategy, n_div = 10): 
    '''
        return the index of k_best. Because the strategy uses name_to_index, they are correctly indexed.
    '''

    k_best, distances = strategy.find_best(tile_list, get_dist=True) #[(name, average_tile)]

    return k_best, distances

def init_name_dic(max):
    dirs = os.listdir("dataset/")
    dic = {}
    
    i=0
    for name in dirs:
        if i>=max:
            break
        dic[name]=i
        i += 1
    
    return dic

class ParrMemory:
    def __init__(self, genq=None, resq=None):
        '''
            n_tiles: number of tiles to be generated horizontally and vertically
            name: 
        '''
        self.genq = genq
        self.resq = resq
    
    def add_new(self, strategy):
        tile = False
        while tile is False:
            tile = self.genq.get()
            tile_list = self.group_transform(tile)

        labels, distances = generate_label(tile_list, strategy)
        self.resq.put((tile, labels, distances) )
    
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
        

def write_queue(resq, file_name):
    i=0
    while True:
        with open(f"{file_name}{i}.pkl", "wb") as f:
            image, labels, distances = resq.get()
            pickle.dump((image, labels, distances), f, -1)
            i+=1
        if (i+1)%500==0:
            print(i)

def gen_tiles(genq):
    '''
        Puts tiles in a genq list
    '''
    path = "train_data/"
    l = os.listdir(path)
    for name in l:
        with open(os.path.join(path, name), 'rb') as f:
            tile = pickle.load(f)[0]
        
        genq.put(tile)

def generate_n(resq, genq, n, strategy, name_to_index, use_gpu):
    pprint("Generating strategy object..")
    strategy_obj = strategy(name_to_index, limit=None, use_cells=True)
    pprint("Done generating strategy object.")
    memory = ParrMemory(genq, resq)
    for i in range(n):
        t = time.time()
        memory.add_new(strategy_obj)
        pprint(time.time()-t)
        if i%5==4:
            pprint(f"{i+1}/{n}")
        

if __name__ == "__main__":
    num_proc = 1

    strategy = AverageStrategyCosineFaiss
    num_gen_per_proc = 10000

    with open("name_to_index.pkl", "rb") as f:
        name_to_index = pickle.load(f)
    
    m = mp.Manager()
    resq = m.Queue(100)
    genq = m.Queue(16)

    gen_process = mp.Process(target=gen_tiles, args=(genq,), name="Generator")
    write_process = mp.Process(target=write_queue, args=(resq, "train_data_augmented/data_"), name="Writer")

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