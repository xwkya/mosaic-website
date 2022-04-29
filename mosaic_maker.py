from re import L
from correctors import Combiner, Corrector, HSICorrector, LinearCorrector, AffineCorrector
from nn_models import NNpolicy_torchresize
from strategies import AverageStrategy, AverageStrategyCosine, AverageStrategyCosineFaiss, AverageXLuminosity, NNStrategy
from collections import defaultdict
from helper_func import print_parameters
import numpy as np
import cv2
import time
import os
import pickle
import math
import skimage.measure
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

image_name = "image2.jpeg"
limit = 1
num_tiles = 48
search_rotations = True
search_symmetry = True
upsize_depth_search = 2
quality = True
strategy_name = 'Faiss'
sample_network = False
sample_temperature = 1.3
upsize_discount = 0.8 # Allow the upsize discount to be x% worse than the small tiles


parameters = {
    "limit": limit,
    "n_tiles": num_tiles,
    "search_rotations": search_rotations,
    "search_symmetry": search_symmetry,
    "upsize_depth_search": upsize_depth_search,
    "quality": quality,
    'strategy': strategy_name,
    'sample': sample_network,
    'sample_temp': sample_temperature,
    'upsize_discount': upsize_discount
}

def save_mosaic(strategy, parameters, save_name, mosaic, folder = "mosaics"):
    if parameters["limit"] is None:
        cv2.imwrite(folder + "/" + str(parameters["n_tiles"]) + "_" + str(strategy.max) + "_"+save_name, mosaic)
    else:
        cv2.imwrite(folder + "/" + str(parameters["n_tiles"] ) +"_" +str(parameters["limit"]) + "_" + save_name, mosaic)

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

def best_to_mosaic(n_tiles_w, n_tiles_l, tile_size, image, best, parameters, corrector, strategy):
    upsize_discount = parameters['upsize_discount']
    inverse_rotation = get_inverse_rotation_dic()

    if parameters["quality"]:
        mosaic = np.zeros(shape=(n_tiles_w*256,n_tiles_l*256,3))
    else:
        mosaic = np.zeros_like(image)


    score_dict = {}
    # Run for each depth of upsize
    for upsize_depth in range(1, parameters["upsize_depth_search"]+1):
        for i in range(n_tiles_w - upsize_depth + 1):
            for j in range(n_tiles_l - upsize_depth +1):
                
                place_tile = False # Flag to determine whether or not to place the tile

                src_tile_resized = cv2.resize(image[i*tile_size : (i+upsize_depth)*tile_size, j*tile_size : (j+upsize_depth)*tile_size, :], (256,256))
                name, score, infos = best[(i, j, i+upsize_depth, j+upsize_depth)]

                if upsize_depth == 1:
                    place_tile = True
                    score_dict[(i,j)] = score
                
                else:
                    # Test score to determine if we place the tile
                    avg_score_subtile = 0
                    for si in range(i, i+upsize_depth):
                        for sj in range(j, j+upsize_depth):
                            avg_score_subtile += score_dict[(si, sj)]
                    
                    avg_score_subtile /= upsize_depth**2
                    if score < upsize_discount * avg_score_subtile:
                        place_tile = True

                if not place_tile:
                    continue

                replacement = cv2.imread("dataset/"+name)
                
                # infos contains the rotation applied to the src tile -> inverse rotation must be applied to the replacement
                if parameters["search_symmetry"]:
                    if infos["symmetry"] == -1:
                        replacement = cv2.flip(replacement, 0)

                if parameters["search_rotations"]:
                    if infos["rotation"] is not None:
                        replacement = cv2.rotate(replacement, inverse_rotation[infos["rotation"]])
                
                # Update the scores

                if upsize_depth > 1:
                    for si in range(i, i+upsize_depth):
                        for sj in range(j, j+upsize_depth):
                            sub_replacement = replacement[((si-i)*256)//upsize_depth: (si+1-i)*256//upsize_depth, ((sj-j)*256)//upsize_depth: (sj+1-j)*256//upsize_depth, :]
                            sub_src_tile = src_tile_resized[((si-i)*256)//upsize_depth: (si+1-i)*256//upsize_depth, ((sj-j)*256)//upsize_depth: (sj+1-j)*256//upsize_depth, :]
                            sub_score = strategy.find_distance(strategy.average(sub_replacement), strategy.average(sub_src_tile))
                            score_dict[(si, sj)] = sub_score
                
                replacement = corrector.colour(replacement, src_tile_resized)

                if upsize_depth > 1:
                    replacement = cv2.resize(replacement, (256*upsize_depth, 256*upsize_depth))

                if parameters['quality']:
                    mosaic[i*256:(i+upsize_depth)*256, j*256:(j+upsize_depth)*256, :] = replacement
                else:
                    replacement = cv2.resize(replacement, (tile_size*upsize_depth, tile_size*upsize_depth))
                    mosaic[i*tile_size : (i+upsize_depth)*tile_size, j*tile_size : (j+upsize_depth)*tile_size, :] = replacement

    return mosaic, score_dict

def get_inverse_rotation_dic():
    inverse_transform = {cv2.ROTATE_90_CLOCKWISE: cv2.ROTATE_90_COUNTERCLOCKWISE,\
         cv2.ROTATE_90_COUNTERCLOCKWISE: cv2.ROTATE_90_CLOCKWISE,\
         cv2.ROTATE_180: cv2.ROTATE_180}
        
    return inverse_transform

def keep_best(name_replacement_list, tile_list, infos_list, strategy, corrector):

    best = defaultdict(lambda : (None, math.inf, None)) # (name, score, infos)
    for name, tile_ref, infos in zip(name_replacement_list, tile_list, infos_list):
        coords = infos["coords"]
        replacement = strategy.average_map[name]
        
        # No need to rotate back because the tile_list contains the original tile already rotated. Only need to compare replacement to this
        # replacement = corrector.colour(replacement, strategy.average(tile_ref))
        score = strategy.find_distance(replacement, strategy.average(tile_ref))

        if score < best[coords][1]:
            best[coords] = (name, score, infos)
    
    return best

def make_mosaic(image, strategy, corrector, parameters):
    
    response = {}
    w, l = image.shape[0], image.shape[1]

    tile_size = min(w//parameters["n_tiles"], l//parameters["n_tiles"])
    n_tiles_w = w//tile_size
    n_tiles_l = l//tile_size

    tile_infos_list = []
    for upsize_depth in range(1, parameters["upsize_depth_search"]+1):
        x = image_splitter(image, n_tiles_w, n_tiles_l, tile_size, upsize_depth=upsize_depth)
        if parameters["search_rotations"]:
            x = rotation_generator(x)
        
        if parameters["search_symmetry"]:
            x = symmetry_generator(x)
        
        tile_infos_list += x

    # tile_infos_list contains (tile, infos), infos = {"coords": coords, "rotation": rotation, transformation_name: transformation}

    tile_list = [x[0] for x in tile_infos_list]
    infos_list = [x[1] for x in tile_infos_list]
    
    s = time.time()
    name_replacement_list = strategy.find_best_n(tile_list)
    print("find best n:",time.time()-s)

    s = time.time()
    best = keep_best(name_replacement_list, tile_list, infos_list, strategy, corrector)
    print("keep best:",time.time()-s)

    s = time.time()
    mosaic, score_dict = best_to_mosaic(n_tiles_w, n_tiles_l, tile_size, image, best, parameters, corrector, strategy)
    print("best to mosaic:",time.time()-s)
    
    response["mosaic"] = mosaic
    response["score_dict"] = score_dict
    
    return response


# returns [(tile, infos)]
# infos = {coords: (x1,y1,x2,y2))}
def image_splitter(image, n_tiles_w, n_tiles_l, tile_size, upsize_depth=1):
    tile_list = []

    for i in range(n_tiles_w - upsize_depth + 1):
        for j in range(n_tiles_l - upsize_depth + 1):
            infos = {"coords":(i,j,i+upsize_depth,j+upsize_depth)}

            tile = image[i*tile_size : (i+upsize_depth)*tile_size, j*tile_size : (j+upsize_depth)*tile_size, :]
            tile = skimage.measure.block_reduce(tile, (upsize_depth, upsize_depth, 1), np.mean)
            tile_list.append((tile, infos))
    
    return tile_list

# returns [(tile, infos)]
# adds rotation
def rotation_generator(tile_list):
    new_tile_list = []

    for tile, infos in tile_list:
        new_infos = {key: infos[key] for key in infos}
        new_infos["rotation"] = None
        new_tile_list.append((tile, new_infos))
        
        for rotation in [cv2.ROTATE_90_CLOCKWISE,cv2.ROTATE_180,cv2.ROTATE_90_COUNTERCLOCKWISE]:
            new_infos = {key: infos[key] for key in infos}
            new_infos["rotation"] = rotation
            new_tile_list.append((cv2.rotate(tile, rotation), new_infos))
    
    return new_tile_list

def symmetry_generator(tile_list):
    new_tile_list = []
    for tile, infos in tile_list:
        for symmetry in [1,-1]:
            new_infos = {key: infos[key] for key in infos}
            new_infos["symmetry"] = symmetry
            if symmetry == -1:
                new_tile_list.append((cv2.flip(tile, 0), new_infos))
            else:
                new_tile_list.append((tile, new_infos))
    
    return new_tile_list

if __name__ == '__main__':
    image = cv2.imread(image_name)

    print_parameters(parameters)

    with open("name_to_index.pkl", "rb") as f:
        name_to_index = pickle.load(f)

    
    print("Initializing strategy object..")

    if parameters['strategy'] == 'NN':
        NN = NNpolicy_torchresize(10000, name_to_index, "cosine")
        NN.load()
        strategy = NNStrategy(NN, True, max=parameters['limit'], sample=parameters['sample'], sample_temp=parameters['sample_temp'])
    
    elif parameters['strategy'] == 'Faiss':
        strategy = AverageStrategyCosineFaiss(name_to_index, limit=parameters['limit'], use_cells=False)
    
    print("Done generating strategy object.")

    corrector = AffineCorrector(max_affine=8, max_linear=30)

    
    x = make_mosaic(image, strategy, corrector, parameters)
        
    save_mosaic(strategy, parameters, "paul.jpeg", x["mosaic"], "mosaics/NN_cosine")
