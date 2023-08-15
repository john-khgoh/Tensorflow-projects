import tensorflow as tf
import zipfile
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt

#Getting the image directories
cwd = os.getcwd()
train = cwd + '\\tmp\\Covid19-dataset\\train'
test = cwd + '\\tmp\\Covid19-dataset\\test'

'''
os.system('wget --no-check-certificate \
https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 \
-O %s\\tmp\\inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5' %cwd)
'''

local_weight_file = cwd + '\\tmp\\inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(input_shape=(150,150,3),
                                include_top=False,
                                weights=local_weight_file)

#Setting the layers to non-trainable
for layer in pre_trained_model.layers:
    layer.trainable = False

#print(pre_trained_model.summary())

last_layer = pre_trained_model.get_layer('mixed7')
last_output = last_layer.output

x = layers.Flatten()(last_output)
x = layers.Dense(1024,activation='relu')(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(3,activation='softmax')(x)
model = Model(pre_trained_model.input,x)
model.summary()

model.compile(optimizer=Adam(learning_rate=1e-4),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   zoom_range = 0.1)
test_datagen = ImageDataGenerator(rescale= 1./255.)
train_generator = train_datagen.flow_from_directory(train,
                                                    batch_size=10,
                                                    class_mode='categorical',
                                                    target_size=(150,150))
test_generator = test_datagen.flow_from_directory(test,
                                                  batch_size=10,
                                                  class_mode='categorical',
                                                  target_size=(150,150))

history = model.fit(train_generator,
                    validation_data=test_generator,
                    steps_per_epoch=10,
                    epochs=10,
                    validation_steps=5,
                    verbose=True)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs,acc,'r',label='Training_accuracy')
plt.plot(epochs,val_acc,'b',label='Test_accuracy')
plt.title('Training and test accuracy')
plt.figure()
plt.show()