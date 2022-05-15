from general_trainer import LitModel
import torch
import cv2
import pickle
import numpy as np

model = LitModel(NN_name='GCNN', load=False)
model = model.load_from_checkpoint('lightning_logs/version_3/checkpoints/epoch=20-step=14091.ckpt', NN_name='GCNN', load=False)

x = cv2.resize(cv2.imread('subimage_2.jpeg'),(8,8))

y = torch.argmax(model(torch.Tensor(x).unsqueeze(0))[0],dim=-1).squeeze()
with open('name_to_index.pkl', 'rb') as f:
    name_to_index = pickle.load(f)

index_to_name = {name_to_index[k]: k for k in name_to_index}

for i in range(8):
    rs = np.zeros((256*2, 256*4,3))
    r = cv2.imread('dataset_r/'+index_to_name[y[i].item()])
    cv2.imshow(f'r{i}',r)
    if i<4:
        rs[:256, 256*i:256*(i+1), :] = r
    else:
        rs[256:, 256*(i-4):256*(i-4+1), :] = r

cv2.waitKey()