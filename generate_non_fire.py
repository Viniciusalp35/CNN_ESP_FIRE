import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import save_img
import os


path = 'data/non_fire'
TO_ADD = 600

print("Baixando CIFAR-10...")

(x_train,_) , (_,_) = tf.keras.datasets.cifar10.load_data()

all_images = x_train

indexes = np.random.choice(len(all_images),TO_ADD,replace=False)

print(f"saving {TO_ADD} random images in {path}...")

for i, idx in enumerate(indexes):
    img = all_images[idx]
    file_name = f'cifar10_img_{i}.jpg'
    complete_file_name = os.path.join(path,file_name)

    save_img(complete_file_name,img)