import random
from image_generator import get_image, img_to_tiles
import pickle
import multiprocessing as mp

i = 0
j = 0

def dataset_reader(read_q):
    i=0
    with open('dataset.pkl', "rb") as conn:
        while True:
            i+=1
            if i%100==0:
                print(i)
            try:
                url, labels = pickle.load(conn)
                read_q.put((url, labels))
            except EOFError:
                print('Done')
                break
    
    while True:
        pass

def download_splitter(read_q, i):
    while True:
        url, labels = read_q.get()
        image = get_image(url, f'{i}')
        if image is False:
            continue
        
        tile_list = img_to_tiles(image, 32)
        try:
            assert len(tile_list) == len(labels)
        except AssertionError:
            print(len(tile_list), len(labels))
            continue
        
        i=0
        for tile, label in zip(tile_list, labels):
            with open(f'train_data/{url.replace("/", "")}_{i}.pkl', 'wb') as f:
                pickle.dump((tile, label), f)
            i+=1
        


if __name__ == '__main__':
    num_downloader = 32
    manager = mp.Manager()
    read_q = manager.Queue(4*num_downloader)
    reader_proc = mp.Process(target=dataset_reader, args=(read_q,), name="Dataset Reader")
    down_procs = []
    for i in range(num_downloader):
        down_procs.append(mp.Process(target=download_splitter, args=(read_q, i), name=f"Dowloader {i}"))
    

    reader_proc.start()
    
    print('Generating generator processes')
    for proc in down_procs:
        proc.start()
    

    reader_proc.join()

        
