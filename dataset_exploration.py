import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from image_generator import get_image

def dataset_statistics(num_iter, display_images=False):
    with open("name_to_index10000.pkl", "rb") as f:
        name_to_index = pickle.load(f)
    
    index_to_name = {v: k for k, v in name_to_index.items()}

    with open("dataset.pkl", "rb") as f:
        x = np.zeros((10000,))
        for i in range(num_iter):
            try:
                url, label_list = pickle.load(f)
                x = x + np.eye(10000)[np.array(label_list)].sum(axis=(0,1))
            except EOFError:
                break

    ind = np.argpartition(x, -12)[-12:]
    print("Favourite images index:", ind)
    print("These are their values:", x[ind])
    print("This is the sum of all the values:",np.sum(x))
    if not display_images:
        return
    
    rows, columns = 3, 4
    fig = plt.figure(figsize=(8, 8))

    for i in range(1, columns*rows +1):
        img = cv2.imread('dataset/'+index_to_name[ind[i-1]])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax = fig.add_subplot(rows, columns, i)
        ax.set_title(f"{x[ind[i-1]]} occurence")
        plt.axis('off')
        plt.imshow(img)
    
    plt.show()

def make_mosaic(url, labels_list, index_to_name):
    img = get_image(url)
    w, l = img.shape[0], img.shape[1]

    tile_size = min(w//32, l//32)
    n_tiles_w = w//tile_size
    n_tiles_l = l//tile_size

    mosaic = np.zeros(shape=(n_tiles_w*256,n_tiles_l*256,3))

    for i in range(n_tiles_w):
        for j in range(n_tiles_l):
            name_replacement = index_to_name[labels_list[i * n_tiles_l + j]]
            replacement = cv2.imread("dataset/"+name_replacement)
            #replacement = corrector.colour(replacement, cv2.resize(tile_img, (256, 256)))
            mosaic[i*256:(i+1)*256, j*256:(j+1)*256, :] = replacement
    
    return img, mosaic


with open("name_to_index.pkl", "rb") as f:
    name_to_index = pickle.load(f)
    
index_to_name = {v: k for k, v in name_to_index.items()}

with open("dataset.pkl", 'rb') as f:
    for _ in range(47):
        pickle.load(f)
    url, labels_list = pickle.load(f)
    img, mosaic = make_mosaic(url, labels_list, index_to_name)
    cv2.imwrite("test.jpg",mosaic)
    cv2.imwrite("test2.jpg", img)
    #plt.imshow(cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))
    #plt.show()

    

dataset_statistics(1000, display_images=True)


