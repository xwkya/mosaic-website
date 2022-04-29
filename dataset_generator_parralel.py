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

    k_best = strategy.find_best(tile_list) #[(name, average_tile)]

    return k_best

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
    def __init__(self, n_tiles, genq=None, resq=None):
        '''
            n_tiles: number of tiles to be generated horizontally and vertically
            name: 
        '''
        self.n_tiles = n_tiles
        self.genq = genq
        self.resq = resq
    
    def add_new(self, strategy):
        tile_list = False
        while type(tile_list) == type(False):
            url, tile_list = self.genq.get()
        p_list = generate_label(tile_list, strategy)
        self.resq.put( (url, p_list) )

def write_queue(resq, file_name):
    with open(file_name, "wb") as f:
        while True:
            url, labels = resq.get()
            pickle.dump((url, labels), f, -1)

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
        
        if type(img)==type(False):
            continue
        tile_list = img_to_tiles(img, n_tiles)
        genq.put((url, tile_list))

def generate_n(n_tiles, resq, genq, n, strategy, name_to_index, use_gpu):
    pprint("Generating strategy object..")
    strategy_obj = strategy('dataset/', name_to_index, use_gpu)
    pprint("Done generating strategy object.")
    memory = ParrMemory(n_tiles, genq, resq)
    for i in range(n):
        t = time.time()
        memory.add_new(strategy_obj)
        pprint(time.time()-t)
        if i%5==4:
            pprint(f"{i+1}/{n}")
        

if __name__ == "__main__":
    n_tiles = 24
    name_to_index = init_name_dic(10000)
    num_proc = 1

    strategy = AverageStrategyCosineFaiss
    num_gen_per_proc = 10000

    with open("name_to_index.pkl", "wb") as f:
        pickle.dump(name_to_index, f, protocol=-1)
    
    m = mp.Manager()
    resq = m.Queue(100)
    genq = m.Queue(16)

    gen_process = mp.Process(target=gen_tiles, args=(genq, n_tiles), name="Generator")
    write_process = mp.Process(target=write_queue, args=(resq, "dataset.pkl"), name="Writer")

    procs = []
    for i in range(num_proc):
        procs.append(mp.Process(target=generate_n, args=(n_tiles, resq, genq, num_gen_per_proc, strategy, name_to_index, False), name=f"Tiling {i+1}"))

    print("Starting Processes..")
    gen_process.start()

    for proc in procs:
        proc.start()

    write_process.start()

    print("Finished starting processes..")


    gen_process.join()