import os
import numpy as np
import sys

import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize)
cwd = os.getcwd()
checkpoint_filepath = cwd + '\\venv\\checkpoint\\'

#Loading the data from TFDS
train_ds= tfds.load('malaria',split=['train[:100%]'])
                                 #as_supervised=True,
                                 #with_info=True)
#no_of_classes = info.features['label'].num_classes

data = []
labels = []
cnt = 0 

for i in train_ds[0]:
    #data.append(tf.cast(i['image'],dtype=tf.float32)/255.0)
    data.append(i['image'])
    labels.append(i['label'])

#Train-validation split
total_len = len(train_ds[0])
train_pct = 0.8
train_len = int(train_pct * total_len)

train_data = np.array(data[:train_len],dtype=object)
train_label = np.array(labels[:train_len],dtype=object) 
valid_data = np.array(data[train_len:],dtype=object)
valid_label = np.array(data[:train_len],dtype=object)  

