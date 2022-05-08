import random
from image_generator import get_image

def get_random_image(size):
    filesize = 99000                 #size of the really big file
    offset = random.randrange(filesize)

    f = open('train_images.tsv')
    image = False
    while image is False:
        f.seek(offset)                  #go to random position
        f.readline()                    # discard - bound to be partial line
        random_line = f.readline()      # bingo!

        # extra to handle last/first line edge cases
        if len(random_line) == 0:       # we have hit the end
            f.seek(0)
            random_line = f.readline()
        

        image = get_image(random_line.split('\t')[0])
        if image is False:
            continue

        h, w, c = image.shape
        if h<size or w<size:
            image = False
            continue
    
    y, x = random.randint(0, h-size-1), random.randint(0, w-size-1)
    return image[y:y+size, x:x+size, :]
