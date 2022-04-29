from trainer_queue import start_training
import pickle
from nn_models import NNConstructed
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def grid_search(norm2d_list, norm1d_list, conv_channels_list, linear_channels_list, generate_n):
    
    with open("name_to_index.pkl", 'rb') as f:
        name_to_index = pickle.load(f)

    NN_class = NNConstructed
    num_generators = 6

    for norm2d in norm2d_list:
        for norm1d in norm1d_list:
            for index_conv, conv_channels in enumerate(conv_channels_list):
                for index_lin, linear_channels in enumerate(linear_channels_list):

                    name = str(norm2d)+"_"+str(norm1d)+"_"+str(index_lin)+"_"+str(index_conv)+"_cosine"
                    args = (10000, name_to_index, name, norm2d, norm1d, linear_channels, conv_channels)

                    print(f"---- Starting training with parameters: {name} -----")

                    loss_list = start_training(NN_class, args, num_generators, generate_n)
                    with open("losses/loss_"+name+".pkl", "wb") as f:
                        pickle.dump(loss_list, f)
                    
                    print("Mean loss:",np.mean(loss_list))
                    print("Last 200 epochs loss:",np.mean(loss_list[-200:]))


if __name__ == '__main__':

    norm1d_list = [True, False]
    norm2d_list = [True, False]
    conv_channels_list = [[32, 64, 128], [16, 32, 64], [32, 64], [64, 128, 128]]
    linear_channels_list = [[1024], [2048], [512], [1024, 1024], [2048, 2048], [1024,1024,1024]]

    grid_search(norm2d_list, norm1d_list, conv_channels_list, linear_channels_list, 2000)

# Best seems to be [64, 128, 128], [2048, 2048]
    
    