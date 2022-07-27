import cv2, os
import numpy as np
import time
import torch
import faiss
import itertools
import pickle
from torchvision import transforms as T
from mosaic_project.NN_training.general_trainer import LitModel


class AverageStrategy:
    def __init__(self, divides, path, max):
        self.path = path
        self.divides = divides
        self.average_map = {}
        self.name = "average"
        self.max = max
        self.init_average(max)

    def init_average(self, max):
        dirs = os.listdir(self.path)
        i=0
        for name in dirs:
            if i>=max:
                break
            img = cv2.imread(self.path + name)
            self.average_map[name] = self.average(img)
            i += 1
        
    def average(self, img):
        average_t = np.zeros(shape=(self.divides, self.divides, 3))
            
        for i in range(self.divides):
            for j in range(self.divides):
                for k in range(3):
                    average_t[i,j,k] = np.mean(img[(i*img.shape[0])//self.divides : ((i+1)*img.shape[0])//self.divides, (j*img.shape[1])//self.divides : ((j+1)*img.shape[1])//self.divides, k])
        
        return average_t

    def find_distance(self, avg1, avg2):
        return np.sum(np.square(avg1-avg2))

    def find_best(self, tile):
        average_tile = self.average(tile)
        best = 9999999999999999
        best_name = None
        for name in self.average_map:
            distance = self.find_distance(self.average_map[name], average_tile)
            if distance < best:
                best = distance
                best_name = name
        
        return best_name
    
    def find_best_n(self, tile_list):
        return [self.find_best(tile) for tile in tile_list]

    def find_k_best(self, tile, k):
        average_tile = self.average(tile)
        h = []
        largest = 999999999999
        
        for name in self.average_map:
            distance = self.find_distance(self.average_map[name], average_tile)

            if distance < largest:
                if len(h)<k:
                    h.append((name, distance))
                else:
                    h[-1] = (name, distance)
                h.sort(key=lambda x: x[1])
                largest = h[-1][1]
        
        return [x for x in h]

class AverageStrategyCosine:
    def __init__(self, path, name_to_index):
        self.name_to_index = name_to_index

        self.path = path
        self.average_map = {}
        self.quantization_table = self.generate_quantization()
        self.name = "cosine"
        self.max = len(name_to_index)
        self.init_average(max)
    
    def average(self, img):
        img = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2YCR_CB)
        average_t = np.zeros(shape=(8, 8, 3))


        # Average over 8 pixels
        for i in range(8):
            for j in range(8):
                for k in range(3):
                    average_t[i,j,k] = np.mean(img[(i*img.shape[0])//8 : ((i+1)*img.shape[0])//8, (j*img.shape[1])//8 : ((j+1)*img.shape[1])//8, k])
        
        # Run the Discrete Cosine Transform on the luminescence
        imf = np.float32(average_t[:,:,0])/255.0  # float conversion/scale
        dct = cv2.dct(imf)              # the dct
        imgcv1 = dct*255.0    # convert back to int
        imgcv1 = imgcv1/self.quantization_table
        average_t[:, :, 0] = imgcv1

        return average_t



    def init_average(self, max):
        for name in self.name_to_index:
            img = cv2.imread(self.path + name)
            self.average_map[name] = self.average(img)

    def find_distance(self, avg1, avg2):
        return np.sum(np.square(avg1[:,:,0]-avg2[:,:,0]))*np.mean(self.quantization_table)+0.2*(np.sum(np.square(avg1-avg2)))
    
    def find_best(self, tile):
        average_tile = self.average(tile)
        best = 9999999999999999
        best_name = None
        for name in self.average_map:
            distance = self.find_distance(self.average_map[name], average_tile)
            if distance < best:
                best = distance
                best_name = name
        
        return [best_name]

    def find_best_n(self, tile_list, limit, search_k):
        return [self.find_best(x) for x in tile_list]

    def find_k_best(self, tile, k):
        average_tile = self.average(tile)
        h = []
        largest = 999999999999
        
        for name in self.average_map:
            distance = self.find_distance(self.average_map[name], average_tile)

            if distance < largest:
                if len(h)<k:
                    h.append((name, distance))
                else:
                    h[-1] = (name, distance)
                h.sort(key=lambda x: x[1])
                largest = h[-1][1]
        
        return [x for x in h]
    
    def find_distribution(self, tile):
        '''
            Return probabilities, replacement_names.
            probabilities: list of probabilities for each tile to be selected
            replacement_names: list of the names of each replacement in the same order as probabilities
        '''
        average_tile = self.average(tile)
        distance_arr = np.array([self.find_distance(self.average_map[name], average_tile) for name in self.average_map])
        name_arr = [name for name in self.average_map]
        distance_arr = distance_arr/np.mean(distance_arr)
        distance_arr_exp = np.exp(-distance_arr)
        
        return distance_arr_exp/np.sum(distance_arr_exp), name_arr
    
    def generate_quantization(self):
        return np.array([[16,11,10,16,24,40,51,61],
                         [12,12,14,19,26,58,60,55],
                         [14,13,16,24,40,57,69,56],
                         [14,17,22,29,51,87,80,62],
                         [18,22,37,56,68,109,103,77],
                         [24,35,55,64,81,104,113,92],
                         [49,64,78,87,103,121,120,101],
                         [72,92,95,98,112,100,103,99]])

class AverageStrategyFaiss:
    def __init__(self, name_to_index, divide=4, use_gpu=False, limit=None, use_cells=True, path='data/dataset_r/'):
        self.index_to_name = {v: k for k, v in name_to_index.items()}
        self.path = path
        self.average_map = {}
        self.name = "faissAverage"
        self.name_to_index = name_to_index
        self.use_cells = use_cells
        self.divide = divide

        if limit is None:
            self.max = len(name_to_index)
        else:
            self.max = limit
        
        self.index = self.init_average(use_gpu)
        

    def average(self, img):
        img = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2YCR_CB)
        average_t = np.zeros(shape=(self.divide, self.divide, 3))


        # Average over 8 pixels
        for i in range(self.divide):
            for j in range(self.divide):
                for k in range(3):
                    average_t[i,j,k] = np.mean(img[(i*img.shape[0])//self.divide : ((i+1)*img.shape[0])//self.divide, (j*img.shape[1])//self.divide : ((j+1)*img.shape[1])//self.divide, k])
        
        return average_t

    def init_average(self, use_gpu):
        t = np.zeros((self.max, self.divide, self.divide, 3))
        c = 0
        for name in self.name_to_index:
            if c == self.max:
                break

            img = cv2.imread(self.path + name)
            img_avg = self.average(img)
            self.average_map[name] = img_avg
            t[self.name_to_index[name], :, :, :] = img_avg

            c += 1
        t = t.reshape(self.max,-1)
        index = faiss.IndexFlatL2(self.divide*self.divide*3)
        
        if use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        if self.use_cells:
            index = faiss.IndexIVFFlat(index, self.divide*self.divide*3, 20)

        index.train(t.astype('float32'))
        index.add(t.astype('float32'))

        return index

    def find_distance(self, avg1, avg2):
        return np.sum(np.square(avg1-avg2))
    
    def find_best(self, tile_list, get_dist=False):
        average_tile = [np.ndarray.flatten(self.average(tile)) for tile in tile_list]
        average_tile = np.vstack(average_tile)
        D,I = self.index.search(average_tile.astype('float32'), 1)
        
        if get_dist:
            return I.reshape((-1,)), D.reshape((-1,))

        return [x[0] for x in I]
    
    def find_best_n(self, tile_list):
        indexes = self.find_best(tile_list)
        return [self.index_to_name[ind] for ind in indexes]

class AverageStrategyCosineFaiss:
    def __init__(self, name_to_index, use_gpu=False, limit=None, use_cells=True, scaling=0.45, path='data/dataset_r/'):
        self.index_to_name = {v: k for k, v in name_to_index.items()}
        self.path = path
        self.average_map = {}
        # chr quant gives bad result on l
        self.quantization_table_l = self.generate_quantization_lum()
        self.quantization_table_c = self.generate_quantization_chr()
        self.name = "faiss"
        self.name_to_index = name_to_index
        self.use_cells = use_cells
        self.scaling = scaling

        if limit is None:
            self.max = len(name_to_index)
        else:
            self.max = limit
        
        self.index = self.init_average(use_gpu)
        
    def average(self, img):
        img = np.float32(cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2YCR_CB))
        average_t = np.zeros(shape=(8, 8, 3))

        #print(img[:,:,2])

        # Average over 8 pixels
        for i in range(8):
            for j in range(8):
                for k in range(0,3):
                    average_t[i,j,k] = np.mean(img[(i*img.shape[0])//8 : ((i+1)*img.shape[0])//8, (j*img.shape[1])//8 : ((j+1)*img.shape[1])//8, k])
                #average_t[i,j,0] = np.mean(img[(i*img.shape[0])//8 : ((i+1)*img.shape[0])//8, (j*img.shape[1])//8 : ((j+1)*img.shape[1])//8, 0])
        # Run the Discrete Cosine Transform on every component
        imf = np.float32(average_t[:,:])/255.0  # float conversion/scale
        dct0 = cv2.dct(imf[:, :, 0])*255-128              # the dct
        dct1 = cv2.dct(imf[:, :, 1])*255-128
        dct2 = cv2.dct(imf[:, :, 2])*255-128
        
        dct0 = dct0/self.quantization_table_l
        dct1 = dct1/self.quantization_table_c
        dct2 = dct2/self.quantization_table_c
        average_t[:, :, 0] = dct0
        average_t[:, :, 1] = dct1
        average_t[:, :, 2] = dct2

        #print(np.mean(average_t[:, :, 0]), np.mean(average_t[:, :, 1:]))

        return average_t

    def init_average(self, use_gpu):
        t = np.zeros((self.max, 8, 8, 3))
        c = 0
        for name in self.name_to_index:
            if c == self.max:
                break

            img = cv2.imread(self.path + name)
            img_avg = self.average(img)
            self.average_map[name] = img_avg
            t[self.name_to_index[name], :, :, :] = img_avg

            c += 1
        t = t.reshape(self.max,-1)
        index = faiss.IndexFlatL2(8*8*3)
        
        if use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        if self.use_cells:
            index = faiss.IndexIVFFlat(index, 8*8*3, 20)

        index.train(t.astype('float32'))
        index.add(t.astype('float32'))

        return index

    def find_distance(self, avg1, avg2):
        return np.sum(np.square(avg1-avg2))
    
    def find_best(self, tile_list, get_dist=False):
        average_tile = [np.ndarray.flatten(self.average(tile)) for tile in tile_list]
        average_tile = np.vstack(average_tile)
        D,I = self.index.search(average_tile.astype('float32'), 1)
        
        if get_dist:
            return I.reshape((-1,)), D.reshape((-1,))

        return [x[0] for x in I]
    
    def find_best_n(self, tile_list):
        indexes = self.find_best(tile_list)
        return [self.index_to_name[ind] for ind in indexes]

        

    def find_k_best(self, tile_list, k):
        average_tile = [np.ndarray.flatten(self.average(tile)) for tile in tile_list]
        average_tile = np.vstack(average_tile)
        D,I = self.index.search(average_tile.astype('float32'), k)
        return I
    
    def generate_quantization_lum(self):
        #return np.ones((8,8))
        return np.array([[16,11,10,16,24,40,51,61],
                         [12,12,14,19,26,58,60,55],
                         [14,13,16,24,40,57,69,56],
                         [14,17,22,29,51,87,80,62],
                         [18,22,37,56,68,109,103,77],
                         [24,35,55,64,81,104,113,92],
                         [49,64,78,87,103,121,120,101],
                         [72,92,95,98,112,100,103,99]])

    def generate_quantization_chr(self):
        #return np.ones((8,8))
        return np.array([[16,18,24,47,99,99,99,99],
                         [18,21,26,66,99,99,99,99],
                         [24,26,56,56,99,99,99,99],
                         [47,66,70,99,99,99,99,99],
                         [66,99,99,99,99,99,99,99],
                         [99,99,99,99,99,99,99,99],
                         [99,99,99,99,99,99,99,99],
                         [99,99,99,99,99,99,99,99]])*4

class AverageXLuminosity:
    def __init__(self, divides, path, max):
        self.path = path
        self.divides = divides
        self.average_map = {}
        self.name = "luminMSE"
        self.max = max
        self.init_average(max)

    def init_average(self, max):
        dirs = os.listdir(self.path)
        i=0
        for name in dirs:
            if i>=max:
                break
            img = cv2.imread(self.path + name)
            self.average_map[name] = self.average(img)
            i += 1
        
    def average(self, img):
        average_t = np.zeros(shape=(self.divides, self.divides, 2))
        
        img_hsi = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for i in range(self.divides):
            for j in range(self.divides):
                for k in range(2):
                    average_t[i,j,k] = np.mean(img_hsi[(i*img_hsi.shape[0])//self.divides : ((i+1)*img_hsi.shape[0])//self.divides, (j*img_hsi.shape[1])//self.divides : ((j+1)*img_hsi.shape[1])//self.divides, k])
        
        return average_t

    def find_distance(self, avg1, avg2):
        return np.sum(np.square(avg1-avg2))

    def find_best(self, tile):
        average_tile = self.average(tile)
        best = 9999999999999999
        best_name = None
        for name in self.average_map:
            distance = self.find_distance(self.average_map[name], average_tile)
            if distance < best:
                best = distance
                best_name = name
        
        return best_name
    
    def find_k_best(self, tile, k):
        average_tile = self.average(tile)
        h = []
        largest = 999999999999
        
        for name in self.average_map:
            distance = self.find_distance(self.average_map[name], average_tile)

            if distance < largest:
                if len(h)<k:
                    h.append((name, distance))
                else:
                    h[-1] = (name, distance)
                h.sort(key=lambda x: x[1])
                largest = h[-1][1]
        
        return [x for x in h]

class NNStrategy:
    def __init__(self, litmodel, load, max=None, sample=False, sample_temp = 1.1, pre_path=""):
        self.NN = litmodel
        self.name = "CNN_"+litmodel.NN.name
        self.index_to_name = self.reverse_dic(self.NN.NN.name_to_index)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.NN.to(self.device)
        self.resizer = T.Resize((8,8))
        self.quantization_table = self.generate_quantization()
        self.name_to_index = litmodel.NN.name_to_index
        self.average_map = {}
        self.pre_path = pre_path
        self.path = pre_path+"data/dataset_r/"

        self.sample = sample
        self.sample_temp = sample_temp

        if max is None:
            self.max = len(self.NN.NN.name_to_index)
        else:
            self.max = max
        self.init_average(load)


    def init_average(self, load):
        if not load:
            for name in self.name_to_index:
                img = cv2.imread(self.path + name)
                self.average_map[name] = self.average(img)
            with open(self.pre_path+"average_map.pkl", "wb") as f:
                pickle.dump(self.average_map, f, -1)
        else:
            with open(self.pre_path+"average_map.pkl", "rb") as f:
                self.average_map = pickle.load(f)


    def average(self, img):
        img = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2YCR_CB)
        average_t = np.zeros(shape=(8, 8, 3))


        # Average over 8 pixels
        for i in range(8):
            for j in range(8):
                for k in range(3):
                    average_t[i,j,k] = np.mean(img[(i*img.shape[0])//8 : ((i+1)*img.shape[0])//8, (j*img.shape[1])//8 : ((j+1)*img.shape[1])//8, k]) * 0.45 # Less important.
        
        # Run the Discrete Cosine Transform on the luminescence
        imf = np.float32(average_t[:,:,0])/255.0  # float conversion/scale
        dct = cv2.dct(imf)              # the dct
        imgcv1 = dct*255.0    # convert back to int
        imgcv1 = imgcv1/self.quantization_table
        average_t[:, :, 0] = imgcv1 * np.sqrt(np.mean(self.quantization_table)) # More important

        return average_t

    def find_distance(self, avg1, avg2):
        return np.sum(np.square(avg1[:,:,0]-avg2[:,:,0])) + np.sum(np.square(avg1-avg2)) * 0.5

    def reverse_dic(self, dic):
        # Reverse the keys and values of a dictionary
        d = {}
        for key in dic:
            d[dic[key]] = key
        
        return d
    
    def find_best_n(self, tile_set):
        input = torch.Tensor(np.array(tile_set)).transpose(1,3)
        input = self.resizer(input).to(self.device)/255
        input = input.transpose(1,3)
        #self.NN.eval()
        with torch.no_grad():
            pred = self.NN(input)
        
        pred = pred[:,:self.max]

        if self.sample:
            pred = torch.pow(pred, self.sample_temp)

            pred = pred/torch.sum(pred, dim=-1).unsqueeze(-1)
            cat = torch.distributions.categorical.Categorical(probs=pred)
            replacements_index = cat.sample((1,)).transpose(0,1).squeeze()
        else:
            replacements_index = torch.topk(pred, k=1, dim=1).indices.squeeze()
        
        replacements_names = [self.index_to_name[i.item()] for i in replacements_index]
        
        return replacements_names
    
    def find_best(self, tile):
        return self.find_best_n([tile])[0]

    def generate_quantization(self):
        return np.array([[16,11,10,16,24,40,51,61],
                         [12,12,14,19,26,58,60,55],
                         [14,13,16,24,40,57,69,56],
                         [14,17,22,29,51,87,80,62],
                         [18,22,37,56,68,109,103,77],
                         [24,35,55,64,81,104,113,92],
                         [49,64,78,87,103,121,120,101],
                         [72,92,95,98,112,100,103,99]])

class GNN_strategy:
    def __init__(self, name_to_index):
        model = LitModel(NN_name='GCNN', load=False)
        self.model = model.load_from_checkpoint('lightning_logs/version_6/checkpoints/epoch=40-step=40672.ckpt', NN_name='GCNN', load=False)

        x = cv2.resize(cv2.imread('subimage_2.jpeg'),(8,8))
        self.name_to_index = name_to_index
        self.index_to_name = self.reverse_dic(name_to_index)
        self.average_map = {}
        self.path = "data/dataset_r/"
        self.init_average(True)
        self.resizer = T.Resize((8,8))
        self.max = 10000
        self.quantization_table = np.array([[16,11,10,16,24,40,51,61],
                         [12,12,14,19,26,58,60,55],
                         [14,13,16,24,40,57,69,56],
                         [14,17,22,29,51,87,80,62],
                         [18,22,37,56,68,109,103,77],
                         [24,35,55,64,81,104,113,92],
                         [49,64,78,87,103,121,120,101],
                         [72,92,95,98,112,100,103,99]])
        
    
    def init_average(self, load):
        if not load:
            for name in self.name_to_index:
                img = cv2.imread(self.path + name)
                self.average_map[name] = self.average(img)
            with open("average_map.pkl", "wb") as f:
                pickle.dump(self.average_map, f, -1)
        else:
            with open("average_map.pkl", "rb") as f:
                self.average_map = pickle.load(f)
    
    def reverse_dic(self, dic):
        # Reverse the keys and values of a dictionary
        d = {}
        for key in dic:
            d[dic[key]] = key
        
        return d

    def average(self, img):
        img = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2YCR_CB)
        average_t = np.zeros(shape=(8, 8, 3))


        # Average over 8 pixels
        for i in range(8):
            for j in range(8):
                for k in range(3):
                    average_t[i,j,k] = np.mean(img[(i*img.shape[0])//8 : ((i+1)*img.shape[0])//8, (j*img.shape[1])//8 : ((j+1)*img.shape[1])//8, k]) * 0.45 # Less important.
        
        # Run the Discrete Cosine Transform on the luminescence
        imf = np.float32(average_t[:,:,0])/255.0  # float conversion/scale
        dct = cv2.dct(imf)              # the dct
        imgcv1 = dct*255.0    # convert back to int
        imgcv1 = imgcv1/self.quantization_table
        average_t[:, :, 0] = imgcv1 * np.sqrt(np.mean(self.quantization_table)) # More important

        return average_t

    def find_best_n_trans(self, tile_list):
        with torch.no_grad():
            x = torch.Tensor(np.array(tile_list)).transpose(1,3)
            x = self.resizer(x).transpose(1,3)/255
            self.model.eval()
            y, d = self.model(x)
        
        y = torch.argmax(y,dim=-1)
        best_transform = torch.argmin(d.squeeze(), dim=-1)
        #print(best_transform)
        y = y[range(y.size(0)), best_transform.tolist()]
        names = [self.index_to_name[s.item()] for s in y]
        return names, best_transform.tolist()
    
    def find_distance(self, avg1, avg2):
        return np.sum(np.square(avg1[:,:,0]-avg2[:,:,0])) + np.sum(np.square(avg1-avg2)) * 0.5