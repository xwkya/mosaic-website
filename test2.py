import os 
import cv2
from mosaic_maker_iter import make_mosaic, save_mosaic
from mosaic_evaluators import MosaicEvaluator, resize
from helper_func import print_parameters
from correctors import Combiner, Corrector, HSICorrector, LinearCorrector, AffineCorrector
from nn_models import NNpolicy_torchresize
from strategies import AverageStrategy, AverageStrategyCosine, AverageStrategyCosineFaiss, AverageXLuminosity, NNStrategy
import pickle
import numpy as np

limit = None
num_tiles = 48
search_rotations = True
search_symmetry = True
upsize_depth_search = 2
quality = True
strategy_name = 'NN'
sample_network = False
sample_temperature = 5
upsize_discount = 0.7 # Allow the upsize discount to be x% worse than the small tiles
improve_ratio = 0.2

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

im_dir = "evaluations/images/"
'''
for image_name in os.listdir(im_dir):
    image = cv2.imread(os.path.join(im_dir, image_name))

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
        
    save_mosaic(strategy, parameters, "mosaic_"+image_name, x["mosaic"], "evaluations/mosaics")
'''
evaluator = MosaicEvaluator()
evals = np.zeros((len(os.listdir(im_dir)), len(os.listdir(im_dir))))
for i, image_name in enumerate(os.listdir(im_dir)):
    for j, image_name2 in enumerate(os.listdir(im_dir)):
        img = cv2.imread(os.path.join(im_dir, image_name))
        img = resize(img, 256.5)
        img = cv2.medianBlur(img, 5)

        mosaic = cv2.imread(os.path.join('evaluations/mosaics',"48_10000_mosaic_"+image_name2))
        mosaic = resize(mosaic, 256.5)
        mosaic = cv2.medianBlur(mosaic, 5)


        evals[i, j] = evaluator.evaluate(mosaic, img)[0]

print(evals)

