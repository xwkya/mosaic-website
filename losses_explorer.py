import pickle
import matplotlib.pyplot as plt


norm1d_list = [True]
norm2d_list = [True]
conv_channels_list = [[32, 64, 128], [16, 32, 64], [32, 64], [64, 128, 128]]
linear_channels_list = [[1024], [2048], [512], [1024, 1024], [2048, 2048], [1024,1024,1024]]

for norm2d in norm2d_list:
        for norm1d in norm1d_list:
            for index_conv, conv_channels in enumerate(conv_channels_list):
                for index_lin, linear_channels in enumerate(linear_channels_list):
                    name = str(norm2d)+"_"+str(norm1d)+"_"+str(index_lin)+"_"+str(index_conv)+"_cosine"
                    print("losses/loss_"+name+".pkl")
                    with open("loss_True_True_0_0_cosine.pkl", "rb") as f:
                        h = pickle.load(f)
                    plt.plot(h)
                    plt.show()
                    with open("losses/loss_"+name+".pkl", "rb") as f:
                        h = pickle.load(f)
                    plt.plot(h)
                    plt.show()

