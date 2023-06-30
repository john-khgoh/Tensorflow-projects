#Using a pre-trained model to classify images from tfds

import os
import numpy as np

import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
import matplotlib.pyplot as plt

cwd = os.getcwd()
checkpoint_filepath = cwd + '\\venv\\checkpoint\\'

#Loading the data from TFDS
(train_ds, valid_ds), info = tfds.load('colorectal_histology',
                                 split=['train[:80%]','train[80%:]'],
                                 as_supervised=True,
                                 with_info=True)
no_of_classes = info.features['label'].num_classes

#Data preparation
batch_size = 32
epochs = 15
AUTOTUNE = tf.data.AUTOTUNE
def prepare(ds, shuffle=False):
  # Resize and rescale all datasets.
  ds = ds.map(lambda x, y: (tf.keras.layers.Rescaling(1./255.)(x), y),
              num_parallel_calls=AUTOTUNE)
  ds = ds.cache()
  if shuffle:
    ds = ds.shuffle(1000)
  ds = ds.batch(batch_size)
  return ds.prefetch(buffer_size=AUTOTUNE)

train_ds = prepare(train_ds)
valid_ds = prepare(valid_ds)

#Loading the Inception_v3 weights
# os.system('wget --no-check-certificate \
# https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 \
# -O %s\\inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5' %cwd)
local_weight_file = cwd + '\\inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(input_shape=(150,150,3),
                                include_top=False,
                                weights=local_weight_file)

#Setting the layers to non-trainable
for layer in pre_trained_model.layers:
    layer.trainable = False

last_layer = pre_trained_model.get_layer('mixed7')
#last_output = last_layer.output

#Post pretrained model layers
x = tf.keras.layers.Flatten()(last_layer.output)
x = tf.keras.layers.Dense(1024,activation='relu')(x)
x = tf.keras.layers.Dropout(0.1)(x)
x = tf.keras.layers.Dense(no_of_classes,activation='softmax')(x)
model = Model(pre_trained_model.input,x)
#model.summary()

#Compiling and training the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss='sparse_categorical_crossentropy',
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

