from PIL import Image
import numpy as np
import os

file_list = os.listdir('dataset/TryOn/train/img')

for file_name in file_list:
    file_path = os.path.join('dataset/TryOn/train/img', file_name)
    image = Image.open(file_path)
    image = np.array(image)