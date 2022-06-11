import gym
from gym import spaces
from mosaic_project.image_generator import get_image
import csv
import random
import numpy as np
import cv2

def reverse_dic(self, dic):
    # Reverse the keys and values of a dictionary
    d = {}
    for key in dic:
        d[dic[key]] = key
    
    return d

def get_random_image(size):
    filesize = 99000
    offset = random.randrange(filesize)

    f = open('data/train_images.tsv')
    image = False
    while image is False:
        f.seek(offset)
        f.readline()
        random_line = f.readline()

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

    


class CustomEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, image_gen_id, size, name_to_index):
    super(CustomEnv, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    self.name_to_index = name_to_index
    self.index_to_name = reverse_dic(name_to_index)

    self.size = size
    self.action_space = spaces.MultiDiscrete(nvec=[36, 8, size//2, size//2, len(name_to_index)])
    
    # Example for using image as input:
    # Observation space = (256, 256, 3)^2
    self.observation_space = spaces.Box(low=0, high=255, shape=
                    (size, size, 6), dtype=np.uint8)
    

  def step(self, action):
    # Execute one time step within the environment
    # action = (x, y)
    img = self.target


  def reset(self):
    # Reset the state of the environment to an initial state
    target = get_random_image(size=self.size)
    current = np.zeros((self.size, self.size, 3), dtype=np.uint8)
    self.target = target
    self.construction = current
    return np.concatenate(target, current, axis=-1)

  def render(self, mode='human', close=False):
    # Render the environment to the screen
    cv2.imshow('img', self.target)
    cv2.imshow('img2', self.construction)
    cv2.waitKey(10)