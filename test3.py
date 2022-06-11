import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from skimage import data
from skimage.registration import phase_cross_correlation
from skimage.transform import warp_polar, rotate, rescale
from skimage.util import img_as_float
import random
from scipy.fft import fftn, ifftn, fftfreq
from image_registration.phase_cross_correlation import phase_cross_corr

images = os.listdir('data/dataset_r')

# radius must be large enough to capture useful info in larger image
radius = 128
angle = 73
scale = 1.69
image = cv2.imread('terence.png')
image = cv2.cvtColor(image[256:512, 256:512], cv2.COLOR_BGR2RGB)
image = img_as_float(image)

#image = data.retina()
#image = img_as_float(image[100:-100,100:-100])#[300:800,300:800])
#mask1 = np.zeros_like(image)
#mask1 = cv2.circle(mask1, (image.shape[0]//2,image.shape[1]//2), image.shape[1]//2, (255,255,255), -1)//255
#image = image*mask1

rotated = rotate(image[60:-60,60:-60], angle, resize=False)
rescaled = rescale(rotated, scale, channel_axis=-1)
rescaled = rescaled[20:-20, 20:-20, :]

#rescaled = cv2.imread('data/dataset_r/'+images[random.randint(0,9000)])
#rescaled = cv2.cvtColor(rescaled, cv2.COLOR_BGR2RGB)
#rescaled = img_as_float(rescaled)
#rescaled = rescale(rescaled, .5, channel_axis=-1)

#mask2 = np.zeros_like(rescaled)
#mask2 = cv2.circle(mask2, (rescaled.shape[0]//2,rescaled.shape[1]//2), rescaled.shape[1]//2, (255,255,255), -1)//255
#rescaled = rescaled*mask2


image_polar = warp_polar(image, radius=radius,
                         scaling='log', channel_axis=-1)
rescaled_polar = warp_polar(rescaled, radius=radius,
                            scaling='log', channel_axis=-1)

#shifts, error, phasediff = phase_cross_correlation(image_polar, rescaled_polar,
#                                                   upsample_factor=1)
# Calculate scale factor from translation
#shiftr, shiftc = shifts[:2]
#klog = radius / np.log(radius)
#shift_scale = 1 / (np.exp(shiftc / klog))

shifts = phase_cross_corr(image_polar, rescaled_polar, return_error=False)
shiftr, shiftc = shifts[:2]
print(shifts)

# Calculate scale factor from translation
klog = radius / np.log(radius)
shift_scale = 1 / (np.exp(shiftc / klog))

re_rotated = rotate(rescaled, -shiftr, resize=True, cval=-1)
final_img = rescale(re_rotated, 1/shift_scale, channel_axis=-1, cval=-1)

replacement = np.copy(image)
if final_img.shape[0]>replacement.shape[0]:
    d=final_img.shape[0]-replacement.shape[0]+5
    final_img = final_img[d//2:-d//2, d//2:-d//2]
cx, cy = replacement.shape[0]//2, replacement.shape[1]//2
lx, ly = final_img.shape[0], final_img.shape[1]

mask = (final_img<0).astype(np.int32)
replacement[cx-lx//2: cx+(lx+1)//2, cy-(ly)//2:cy+(ly+1)//2] = final_img *(1-mask) + replacement[cx-lx//2: cx+(lx+1)//2, cy-(ly)//2:cy+(ly+1)//2]*mask

fig, axes = plt.subplots(3, 2, figsize=(10, 10))
ax = axes.ravel()
ax[0].set_title("Original")
ax[0].imshow(image)
ax[1].set_title("Image to match")
ax[1].imshow(rescaled)
ax[2].set_title("Log-Polar-Transformed Original")
ax[2].imshow(image_polar)
ax[3].set_title("Log-Polar-Transformed Image to match")
ax[3].imshow(rescaled_polar)
ax[4].imshow(final_img)
ax[5].imshow(replacement)
plt.show()