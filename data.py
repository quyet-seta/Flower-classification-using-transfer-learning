# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 00:55:06 2021

@author: PC
"""
import os
import tensorflow as tf
from imutils import paths
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import imagenet_utils
from tqdm import tqdm


PATH = 'dataset\\'

def load_data():
    image_paths = list(paths.list_images(PATH))
    random.seed(42)
    random.shuffle(image_paths)
    labels = [p.split(os.path.sep)[-2] for p in image_paths]
    labelEncoder = LabelEncoder()
    labels = labelEncoder.fit_transform(labels)
    
    images = []
    for imagePath in tqdm(image_paths):
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, 0)
        image = imagenet_utils.preprocess_input(image)
        images.append(image)
    images = np.vstack(images)
    
    return (images, labels)

if __name__ == "__main__":
    x,y = load_data()

    