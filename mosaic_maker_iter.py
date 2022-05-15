from correctors import Combiner, Corrector, HSICorrector, LinearCorrector, AffineCorrector
from nn_models import NNpolicy_torchresize
from strategies import AverageStrategyFaiss, AverageStrategyCosine, AverageStrategyCosineFaiss, AverageXLuminosity, NNStrategy, GNN_strategy
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
from mosaic_evaluators import MosaicEvaluator

image_name = "evaluations/images/image2.jpeg"
limit = None
num_tiles = 64
search_rotations = True
search_symmetry = True
upsize_depth_search = 1
quality = True
strategy_name = 'GNN'
sample_network = False
sample_temperature = 5
upsize_discount = 0.7 # Allow the upsize discount to be x% worse than the small tiles
improve_ratio = 0.0

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
    'upsize_discount': upsize_discount,
    'improve_ratio': improve_ratio
}

def save_mosaic(strategy, parameters, save_name, mosaic, folder = "mosaics"):
    if parameters["limit"] is None:
        cv2.imwrite(folder + "/" + str(parameters["n_tiles"]) + "_" + str(strategy.max) + "_"+save_name, mosaic)
    else:
        cv2.imwrite(folder + "/" + str(parameters["n_tiles"] ) +"_" +str(parameters["limit"]) + "_" + save_name, mosaic)

def init_name_dic(max):
    dirs = os.listdir("dataset_r/")
    dic = {}
    
    i=0
    for name in dirs:
        if i>=max:
            break
        dic[name]=i
        i += 1
    
    return dic

def best_to_mosaic(n_tiles_w, n_tiles_l, tile_size, image, best, parameters, corrector, strategy, previous_mos = None, previous_score_dict = None):
    upsize_discount = parameters['upsize_discount']
    inverse_rotation = get_inverse_rotation_dic()

    if previous_mos is not None:
        mosaic = previous_mos

    elif parameters["quality"]:
        mosaic = np.zeros(shape=(n_tiles_w*256,n_tiles_l*256,3))

    else:
        mosaic = np.zeros_like(image)

    if previous_score_dict is not None:
        score_dict = previous_score_dict
    else:
        score_dict = {}

    # Run for each depth of upsize
    for upsize_depth in range(1, parameters["upsize_depth_search"]+1):
        for i in range(n_tiles_w - upsize_depth + 1):
            for j in range(n_tiles_l - upsize_depth +1):
                
                place_tile = False # Flag to determine whether or not to place the tile

                src_tile_resized = cv2.resize(image[i*tile_size : (i+upsize_depth)*tile_size, j*tile_size : (j+upsize_depth)*tile_size, :], (256,256))
                name, score, infos = best[(i, j, i+upsize_depth, j+upsize_depth)]

                if name is None:
                    continue


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

                replacement = cv2.imread("dataset_r/"+name)
                
                # infos contains the rotation applied to the src tile -> inverse rotation must be applied to the replacement
                if parameters['strategy'] != 'GNN':
                    # Base order of operations

                    if "symmetry" in infos:
                        if infos["symmetry"] == -1:
                            replacement = cv2.flip(replacement, 0)

                    if "rotation" in infos:
                        if infos["rotation"] is not None:
                            replacement = cv2.rotate(replacement, inverse_rotation[infos["rotation"]])

                elif parameters['strategy'] == 'GNN':
                    # GNN uses opposite order

                    if infos["rotation"] is not None:
                        replacement = cv2.rotate(replacement, inverse_rotation[infos["rotation"]])

                    if infos["symmetry"] == -1:
                        replacement = cv2.flip(replacement, 0)
                
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

def keep_best(name_replacement_list, tile_list, infos_list, strategy, first_pass = True, improve_indexes = set()):
    best = defaultdict(lambda : (None, math.inf, None)) # (name, score, infos)
    for name, tile_ref, infos in zip(name_replacement_list, tile_list, infos_list):
        coords = infos["coords"]
        
        
        if first_pass and best[coords][0] is not None:
            continue

        if first_pass or (coords[0], coords[1]) in improve_indexes:
            replacement = strategy.average_map[name]
            
            # No need to rotate back because the tile_list contains the original tile already rotated. Only need to compare replacement to this
            score = strategy.find_distance(replacement, strategy.average(tile_ref))

            if score < best[coords][1]:
                best[coords] = (name, score, infos)
            
            continue
    
    return best

def find_worse_score_indexes(score_dict, ratio):
    return set(sorted(list(score_dict.keys()), key=lambda x: score_dict[x])[-int(ratio*len(score_dict)):])

def generate_tile_infos(image, n_tiles_w, n_tiles_l, tile_size, parameters, mask=None):
    tile_infos_list = []
    for upsize_depth in range(1, parameters["upsize_depth_search"]+1):
        x = image_splitter(image, n_tiles_w, n_tiles_l, tile_size, upsize_depth=upsize_depth)
        if mask is not None:
            x = [a for a in x if (a[1]['coords'][0],a[1]['coords'][1]) in mask]

        if parameters["search_rotations"] and parameters['strategy'] != 'GNN':
            x = rotation_generator(x)
        
        if parameters["search_symmetry"] and parameters['strategy'] != 'GNN':
            x = symmetry_generator(x)
        
        tile_infos_list += x
    
    return tile_infos_list

def improve_mosaic(n_tiles_w, n_tiles_l, tile_size, score_dict, previous_mosaic, strategy, corrector, image, parameters):

    # score_dict[coords] = score, coords = (x1, y1)
    improve_indexes = find_worse_score_indexes(score_dict, parameters["improve_ratio"])

    s = time.time()
    new_tile_infos_list = generate_tile_infos(image, n_tiles_w, n_tiles_l, tile_size, parameters, mask=improve_indexes)
    tile_list = [x[0] for x in new_tile_infos_list]
    infos_list = [x[1] for x in new_tile_infos_list]
    name_replacement_list = strategy.find_best_n(tile_list)
    print("find best n:",time.time()-s)

    s = time.time()
    best = keep_best(name_replacement_list, tile_list, infos_list, strategy, first_pass = False, improve_indexes = improve_indexes)
    print("keep best:",time.time()-s)

    s = time.time()
    mosaic, score_dict = best_to_mosaic(n_tiles_w, n_tiles_l, tile_size, image, best, parameters,
        corrector, strategy, previous_mos=previous_mosaic, previous_score_dict=score_dict)
    print("best to mosaic:",time.time()-s)

    return mosaic, score_dict

def make_mosaic(image, strategy, corrector, parameters):
    
    response = {}
    w, l = image.shape[0], image.shape[1]

    tile_size = min(w//parameters["n_tiles"], l//parameters["n_tiles"])
    n_tiles_w = w//tile_size
    n_tiles_l = l//tile_size


    tile_infos_list = image_splitter(image, n_tiles_w, n_tiles_l, tile_size, upsize_depth=1)

    # tile_infos_list contains (tile, infos), infos = {"coords": coords, "rotation": rotation, transformation_name: transformation}

    tile_list = [x[0] for x in tile_infos_list]
    infos_list = [x[1] for x in tile_infos_list]
    

    s = time.time()
    if parameters['strategy']=='GNN':
        # The GNN gives the best rotation/symmetry, we just need to forward that information directly in the infos_list
        name_replacement_list, transforms = strategy.find_best_n_trans(tile_list)
        for x, g in zip(infos_list, transforms):
            x['rotation'] = [None, cv2.ROTATE_90_CLOCKWISE,cv2.ROTATE_180,cv2.ROTATE_90_COUNTERCLOCKWISE][g%4]
            x['symmetry'] = 1 if g<4 else -1

    else:
        name_replacement_list = strategy.find_best_n(tile_list)
    print("find best n:",time.time()-s)

    s = time.time()
    best = keep_best(name_replacement_list, tile_list, infos_list, strategy, first_pass=True)
    print("keep best:",time.time()-s)

    s = time.time()
    mosaic, score_dict = best_to_mosaic(n_tiles_w, n_tiles_l, tile_size, image, best, parameters, corrector, strategy)
    print("best to mosaic:",time.time()-s)

    if improve_ratio > 0.001 and (parameters['search_rotations'] or parameters['search_symmetry']):
        print("improving..")
        mosaic, score_dict = improve_mosaic(n_tiles_w, n_tiles_l, tile_size, score_dict, mosaic, strategy, corrector, image, parameters)
    

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
    ev = MosaicEvaluator()
    print_parameters(parameters)

    with open("name_to_index.pkl", "rb") as f:
        name_to_index = pickle.load(f)

    corrector = AffineCorrector(max_affine=8, max_linear=50)
    
    print("Initializing strategy object..")

    if parameters['strategy'] == 'NN':
        NN = NNpolicy_torchresize(10000, name_to_index, "cosine")
        NN.load()
        strategy = NNStrategy(NN, True, max=parameters['limit'], sample=parameters['sample'], sample_temp=parameters['sample_temp'])
    
    elif parameters['strategy'] == 'FaissCosine':
        strategy = AverageStrategyCosineFaiss(name_to_index, limit=parameters['limit'], use_cells=False, scaling=0.3)
    
    elif parameters['strategy'] == 'average':
        strategy = AverageStrategyFaiss(name_to_index, divide=16)
    
    elif parameters['strategy'] == 'GNN':
        strategy = GNN_strategy(name_to_index)

    x = make_mosaic(image, strategy, corrector, parameters)
    save_mosaic(strategy, parameters, f"GNN.jpeg", x["mosaic"], "mosaics/NN_cosine")
    assert False
    print("Done generating strategy object.")
    
    for scaling in [0.8]:
        print('Scaling:', scaling)
        strategy = AverageStrategyCosineFaiss(name_to_index, limit=parameters['limit'], use_cells=False, scaling=scaling)
        x = make_mosaic(image, strategy, corrector, parameters)
        print('Evaluation:', ev.evaluate(x['mosaic'], image)[0].item())
        print('\n')
        
        save_mosaic(strategy, parameters, f"gays_{scaling}.jpeg", x["mosaic"], "mosaics/NN_cosine")
