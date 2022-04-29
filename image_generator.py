## Importing Necessary Modules
import requests # to get image from the web
import shutil # to save it locally
import csv
import cv2
import pickle

def get_image(url, str=None):
    image_url = url
    if str:
        filename = f"temp_{str}.jpg"
    else:
        filename = "temp.jpg"
    # Open the url image, set stream to True, this will return the stream content.
    r = requests.get(image_url, stream = True)
    # Check if the image was retrieved successfully
    if r.status_code == 200:
        # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
        r.raw.decode_content = True
        
        # Open a local file with wb ( write binary ) permission.
        with open(filename,'wb') as f:
            shutil.copyfileobj(r.raw, f)
    else:
        return False
    
    return cv2.imread(filename)

def img_to_tiles(image, n_tiles):
    w, l = image.shape[0], image.shape[1]
    tile_size = min(w//n_tiles, l//n_tiles)
    n_tiles_w = w//tile_size
    n_tiles_l = l//tile_size
    tiles_list = []
    for i in range(n_tiles_w):
        for j in range(n_tiles_l):
            tile_img = image[i*tile_size : (i+1)*tile_size, j*tile_size : (j+1)*tile_size, :]
            tiles_list.append(cv2.resize(tile_img, (85,85))) # Passing this size to the neural network
            
    return tiles_list