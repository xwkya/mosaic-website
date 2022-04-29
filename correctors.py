from re import L
import cv2
import numpy as np

def clamp(num, min_value, max_value):
   return max(min(num, max_value), min_value)

class Combiner:
    def __init__(self, l):
        self.l = l
        self.name = "_".join([x.name for x in l])

    def colour(self, image, ref):
        for corr in self.l:
            image = corr.colour(image, ref)
        
        return image

class Corrector:
    def __init__(self):
        self.name = "none"

    def colour(self, image, ref):
        return image

class LinearCorrector(Corrector):
    def __init__(self, max_linear):
        super().__init__()
        self.max_linear = max_linear
        self.name = "linear"
    
    def colour(self, image, ref):
        img_init = np.copy(image)
        ref = np.copy(ref).astype(np.int16)
        image = image.astype(np.int16)
        for i in range(3):
            image[:,:,i] = image[:,:,i] + int(clamp(np.mean(ref[:,:,i])-np.mean(image[:,:,i]), (-1)*self.max_linear, self.max_linear))
        
        image = np.clip(image, 0, 255)
        
        #print(np.max(np.abs(image-img_init)))
        return image.astype(np.uint8)

class AffineCorrector(LinearCorrector):
    def __init__(self, max_affine, **kwds):
        super().__init__(**kwds)
        self.max_affine = max_affine
        self.name = "affine"
    
    def colour(self, image, ref):
        image = image.astype(np.int16)
        ref = np.copy(ref).astype(np.int16)
        image = super().colour(image, ref)
        imagec = np.clip(image + np.clip(ref-image, (-1)*self.max_affine, self.max_affine), 0, 255)
        imagec = imagec.astype(np.uint8)
        return imagec

class HSICorrector(Corrector):
    def __init__(self):
        super().__init__()
        self.name = "luminosity"

    def colour(self, image, ref):
        image_hsi = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        ref_hsi = cv2.cvtColor(ref, cv2.COLOR_BGR2HSV)

        image_hsi[:,:,2] = np.clip(np.mean(ref_hsi[:,:,2] - np.mean(image_hsi[:,:,2])) + image_hsi[:,:,2], 1, 254)
        image_bgr = cv2.cvtColor(image_hsi, cv2.COLOR_HSV2BGR)
        return image_bgr