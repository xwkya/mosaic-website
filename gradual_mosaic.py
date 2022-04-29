from correctors import AffineCorrector
from mosaic_evaluators import MosaicEvaluator
from mosaic_maker import make_mosaic, save_mosaic
import cv2
import pickle
import matplotlib.pyplot as plt

from nn_models import NNpolicy_torchresize
from strategies import AverageStrategyCosineFaiss, NNStrategy


image_name = "image2.jpeg"
image = cv2.imread(image_name)


limit = None
num_tiles = 32
search_rotations = True
search_symmetry = True
search_k = 1
upsize_depth_search = 2
requires_evaluation = True
quality = True

parameters = {
    "limit": limit,
    "n_tiles": num_tiles,
    "search_rotations": search_rotations,
    "search_symmetry": search_symmetry,
    "upsize_depth_search": upsize_depth_search,
    "evaluate": requires_evaluation,
    "search_k": search_k,
    "quality": quality
}

# Load the name_to_index
with open("name_to_index.pkl", "rb") as f:
    name_to_index = pickle.load(f)

# Load the NN
print('Loading the neural network if network mosaic is used..')
NN = NNpolicy_torchresize(10000, name_to_index, "cosine")
NN.load()

# Load Strategy and Corrector

#print("Initializing strategy object..")
#strategy = NNStrategy(NN, True)
#print("Done generating strategy object.")

corrector = AffineCorrector(max_affine=8, max_linear=30)

'''
losses = []
for i in range(1,65):
    parameters['n_tiles'] = i
    x = make_mosaic(image, strategy, corrector, parameters)
    if parameters['evaluate']:
        losses.append(network_evaluation(x['mosaic'], image), ssim_evaluation(x['mosaic'], image, True), mixed_evaluation(image, x['mosaic'], i))
    
    save_mosaic(strategy, parameters, "gradual_upsizes.jpeg", x["mosaic"], "mosaics/gradual")
'''
print('Generating evaluator object..')
evaluator = MosaicEvaluator()
losses = []
print('Starting the mosaic generation and evaluation..')
for i in range(100,5000,100):
    parameters['limit'] = i
    strategy = AverageStrategyCosineFaiss('dataset/', name_to_index, limit=parameters['limit'], use_cells=False)
    x = make_mosaic(image, strategy, corrector, parameters)
    if parameters['evaluate']:
        losses.append(evaluator.network_eval(image, x['mosaic']).item())
        print(i, '---', losses[-1])
    
    save_mosaic(strategy, parameters, "datalimit_upsize.jpeg", x["mosaic"], "mosaics/datalimit")

plt.plot(losses, list(range(100,5000,100)))
#plt.show()
plt.savefig('eval_with_limit.png')
with open('losses.pkl', 'wb') as f:
    pickle.dump(losses, f, -1)
