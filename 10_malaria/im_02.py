# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

'''
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
'''
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import os
import numpy as np

import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
import matplotlib.pyplot as plt


cwd = os.getcwd()
checkpoint_filepath = cwd + '/venv/checkpoint/'

#Loading the data from TFDS
(train_ds, valid_ds), info = tfds.load('malaria',
                                 split=['train[:80%]','train[80%:]'],
                                 as_supervised=True,
                                 with_info=True)
no_of_classes = info.features['label'].num_classes

#train_ds = train_ds.as_numpy_iterator()
#Data preparation
batch_size = 64
epochs = 15
AUTOTUNE = tf.data.AUTOTUNE
def prepare(ds,train=False,shuffle=False):
  # Resize and rescale all datasets.
  #ds = ds.map(lambda {x,y}: (tf.keras.layers.Rescaling(1./255.)(x),y),num_parallel_calls=AUTOTUNE)
  ds = ds.map(lambda x,y: (tf.keras.layers.Rescaling(1./255.)(x), y),num_parallel_calls=AUTOTUNE)
  ds = ds.map(lambda x,y: (tf.image.resize(x,(150,150)),y))
  #ds = ds.cache()
  if shuffle:
    ds = ds.shuffle(10000)
  ds = ds.batch(batch_size)
  return ds.prefetch(buffer_size=AUTOTUNE)

train_ds = prepare(train_ds)
valid_ds = prepare(valid_ds)

#Loading the Inception_v3 weights
#os.system('wget --no-check-certificate \
#https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 \
#-O %s\\inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5' %cwd)
#local_weight_file = '/kaggle/input/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
local_weight_file = cwd + '\\inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(input_shape=(150,150,3),
                                include_top=False,
                                weights=local_weight_file)

#Setting the layers to non-trainable
for layer in pre_trained_model.layers:
    layer.trainable = False

last_layer = pre_trained_model.get_layer('mixed7')

#Post pretrained model layers
x = tf.keras.layers.Flatten()(last_layer.output)
x = tf.keras.layers.Dense(4096,activation='relu')(x)
x = tf.keras.layers.Dense(256,activation='relu')(x)
x = tf.keras.layers.Dropout(0.1)(x)
x = tf.keras.layers.Dense(1,activation='sigmoid')(x)
model = Model(pre_trained_model.input,x)

#Compiling and training the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
                loss='binary_crossentropy',
                metrics=['accuracy'])

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

history = model.fit(train_ds,
                    validation_data=valid_ds,
                    epochs=epochs,
                    verbose=True,
                    callbacks=[model_checkpoint_callback])

#Visualizing the accuracy and validation accuracy
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

epochs_l = [*range(epochs)]

plt.plot(epochs_l,acc,'r',label='Training_accuracy')
plt.plot(epochs_l,val_acc,'b',label='Valid_accuracy')
plt.title('Training and validation accuracy')
plt.show()
