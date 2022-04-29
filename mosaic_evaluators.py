import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch
from torchvision import transforms
from PIL import Image
import math
import matplotlib.pyplot as plt

def resize(mosaic, size):
    w, l = mosaic.shape[0], mosaic.shape[1]
    if w<l:
        ratio = size/w
    else:
        ratio = size/l
    
    l = int(ratio*l)
    w = int(ratio*w)
    
    mosaic2 = cv2.resize(mosaic, (w,l), interpolation=cv2.INTER_AREA)
    return mosaic2

class MosaicEvaluator:
    def __init__(self):
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
        self.model.eval()

        self.features = {}

        def get_features(name):
            def hook(model, input, output):
                self.features[name] = output.detach()
            return hook

        self.model.avgpool.register_forward_hook(get_features('feats'))

        self.preprocess = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def ssim_eval(self, mosaic, image):
        
        mosaic = resize(mosaic, 512.5)
        image = resize(image, 512.5)

        mosaic_blurred = cv2.medianBlur(mosaic, 3)
        image_blurred = cv2.medianBlur(image, 3)

        dist = ssim(mosaic_blurred, image_blurred,channel_axis=2)

        return dist*0.8
    
    def network_eval(self, mosaic, img):
        img = img.astype(np.uint8)
        mosaic = mosaic.astype(np.uint8)

        mos_repr = self._get_outputs(mosaic)
        img_repr = self._get_outputs(img)

        return torch.nn.CosineSimilarity(dim=0)(mos_repr, img_repr)
        return 1-torch.sum(torch.square(mos_repr - img_repr)).item()/1200

    def _get_outputs(self, img):
        img = resize(img, 512.5)
        img = cv2.medianBlur(img,3)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_image = Image.fromarray(img)
        input_tensor = self.preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)
        with torch.no_grad():
            self.model(input_batch)

        return self.features['feats'].squeeze()

    def evaluate(self, mosaic, img, n_tiles):
        mosaic = mosaic.astype(np.uint8)
        img = img.astype(np.uint8)
        eval_network = self.network_eval(mosaic, img)
        eval_ssim = self.ssim_eval(mosaic, img)

        imp = math.exp(-n_tiles/12)
        mixed_eval = (1-imp)*eval_network + imp*eval_ssim
        return (mixed_eval, eval_network, eval_ssim)

if __name__ == '__main__':

    evaluator = MosaicEvaluator()

    img = cv2.imread('image2.jpeg')
    img = resize(img, 256.5)
    img = cv2.medianBlur(img,3)
    

    ssim_h, network_h, mixed_h = [], [], []

    for val in range(1,65,4):
        mosaic = resize(cv2.imread(f'mosaics/gradual/{val}_10000_gradual_upsizes.jpeg'),256.5)
        mosaic = cv2.medianBlur(mosaic,3)
        cv2.imwrite('blur_mos.jpeg', mosaic)


        eval = evaluator.evaluate(mosaic, img, val)

        ssim_h.append(eval[2])
        network_h.append(eval[1])
        mixed_h.append(eval[0])

        print(val, '---', eval)

    plt.clf()
    ax = plt.plot(network_h, label='network')
    plt.plot(ssim_h, label='ssim')
    #plt.plot(mixed_h, label='mixed')
    plt.legend()
    plt.show()