import pandas as pd
import numpy as np
from functools import reduce

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_datasets as tfds

from sklearn.base import BaseEstimator, TransformerMixin

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)

(train_ds, valid_ds), info = tfds.load('smartwatch_gestures',
                                       split=['train[:80%]', 'train[80%:]'],
                                       #as_supervised=True,
                                       with_info=True)
no_of_classes = info.features['gesture'].num_classes
#print(no_of_classes)

train_df = tfds.as_dataframe(train_ds)
valid_df = tfds.as_dataframe(train_ds)

max_len = 0
for i in range(len(train_df)):
    temp_len = len(train_df['features/accel_x'][i])
    if(temp_len>max_len):
        max_len = temp_len
for i in range(len(valid_df)):
    temp_len = len(valid_df['features/accel_x'][i])
    if(temp_len>max_len):
        max_len = temp_len

#Padding the values
def padding(col):
    col = col.apply(lambda x:np.pad(x,(max_len-len(x),0),'constant',constant_values=0))
    return col

train_df['features/accel_x'] = padding(train_df['features/accel_x'])
train_df['features/accel_y'] = padding(train_df['features/accel_y'])
train_df['features/accel_z'] = padding(train_df['features/accel_z'])

valid_df['features/accel_x'] = padding(valid_df['features/accel_x'])
valid_df['features/accel_y'] = padding(valid_df['features/accel_y'])
valid_df['features/accel_z'] = padding(valid_df['features/accel_z'])

#Converting the padded values into columns of x, y, z
def df_list_to_columns(col,no_of_col):
    col = pd.DataFrame(col.to_list(),columns=[i for i in range(0,no_of_col)])
    return col

train_df_x = df_list_to_columns(train_df['features/accel_x'],max_len)
train_df_y = df_list_to_columns(train_df['features/accel_y'],max_len)
train_df_z = df_list_to_columns(train_df['features/accel_z'],max_len)

valid_df_x = df_list_to_columns(valid_df['features/accel_x'],max_len)
valid_df_y = df_list_to_columns(valid_df['features/accel_y'],max_len)
valid_df_z = df_list_to_columns(valid_df['features/accel_z'],max_len)

#Concatenating x,y and z
train_x = pd.concat([train_df_x,train_df_y,train_df_z],axis=1)
valid_x = pd.concat([valid_df_x,valid_df_y,valid_df_z],axis=1)

train_y = train_df['gesture']
valid_y = valid_df['gesture']

epochs = 20
model = Sequential([
    tf.keras.layers.Dense(200,activation='relu'),
    tf.keras.layers.Dense(100,activation='relu'),
    tf.keras.layers.Dense(50,activation='relu'),
    tf.keras.layers.Dense(no_of_classes,activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4),
    metrics = ['accuracy']
)

history = model.fit(train_x,train_y,epochs=epochs,validation_data=(valid_x,valid_y))