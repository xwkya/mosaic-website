import os 
import cv2
from NN_training.general_trainer import LitModel
from mosaic_maker_iter import make_mosaic, save_mosaic
from mosaic_evaluators import MosaicEvaluator, resize
from helper_func import print_parameters
from correctors import Combiner, Corrector, HSICorrector, LinearCorrector, AffineCorrector
from nn_models import NNpolicy_torchresize
from strategies import AverageStrategy, AverageStrategyCosine, AverageStrategyCosineFaiss, AverageStrategyFaiss, AverageXLuminosity, NNStrategy
import pickle
import numpy as np
import time

limit = None
num_tiles = 48
search_rotations = True
search_symmetry = True
upsize_depth_search = 1
quality = False
strategy_name = 'NN'
sample_network = False
sample_temperature = 5
upsize_discount = 0.7 # Allow the upsize discount to be x% worse than the small tiles
improve_ratio = 0.2
path = ''

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
    'improve_ratio': improve_ratio,
    'path': path,
}


evaluator = MosaicEvaluator()
im_dir = "evaluations/images/"
with open("name_to_index.pkl", "rb") as f:
    name_to_index = pickle.load(f)

print("Initializing strategy objects..")
NN = LitModel.load_from_checkpoint('models/version_13/checkpoints/epoch=79-step=158720.ckpt', NN_name='CNN', load=False)
NN_strategy = NNStrategy(NN, True, max=parameters['limit'], sample=parameters['sample'], sample_temp=parameters['sample_temp'])
faiss_strategy = AverageStrategyCosineFaiss(name_to_index, limit=parameters['limit'], use_cells=False)

average_strategy = AverageStrategyFaiss(name_to_index, divide=4, use_gpu=False, limit= None, use_cells=False)
print("Done generating strategy objects.")

res = {}

for strategy_name in ['NN', 'FaissCosine', 'Average']:
    for num_tiles in [16,24,32,48,64,72]:
        for improve_ratio in [0., 0.6]:
            for image_name in os.listdir(im_dir):
                parameters['improve_ratio'] = improve_ratio
                parameters['strategy'] = strategy_name
                parameters['n_tiles'] = num_tiles
                
                image = cv2.imread(os.path.join(im_dir, image_name))

                print_parameters(parameters)

                with open("name_to_index.pkl", "rb") as f:
                    name_to_index = pickle.load(f)

                if parameters['strategy'] == 'NN':
                    strategy = NN_strategy
                
                elif parameters['strategy'] == 'FaissCosine':
                    strategy = faiss_strategy
                
                elif parameters['strategy'] == 'Average':
                    strategy = average_strategy
                
                corrector = AffineCorrector(max_affine=8, max_linear=40)

                start=time.time()
                x = make_mosaic(image, strategy, corrector, parameters)
                end = time.time()

                #save_mosaic(strategy, parameters, "mosaic_"+image_name, x["mosaic"], "evaluations/mosaics")

                eval = evaluator.evaluate(x['mosaic'], image)
                print('Evaluation:', round(eval[0].item(),5))
                print("\n")
                res[(image_name, num_tiles, strategy_name, improve_ratio)] = (eval, end-start)


with open('evaluations/res.pkl', 'wb') as f:
    pickle.dump(res, f, -1)


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
'''