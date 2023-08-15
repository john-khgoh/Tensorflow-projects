import keras
import tensorflow as tf
import tensorflow_datasets as tfds
#from tensorflow.keras.preprocessing.text import Tokenizer
#from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
from string import punctuation
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 100)

punctuations = list(punctuation)
tfds.display_progress_bar(enable=True)

##Getting the dataset
ds, info = tfds.load('ag_news_subset', split='train', with_info=True)
dataset_size = info.splits['train'].num_examples  # 120,000 rows

# Convert the TFDS dataset to pandas dataframe for preprocessing
df = tfds.as_dataframe(ds)
# label_list = list(set(list(df['label'])))
df['description'] = df['description'].apply(lambda x: x.decode('utf-8'))
df['title'] = df['title'].apply(lambda x: x.decode('utf-8'))

##Preprocessing
# Removing special characters
df['description'] = df['description'].apply(lambda x: ''.join(i for i in x if not i in punctuations))
df['title'] = df['title'].apply(lambda x: ''.join(i for i in x if not i in punctuations))

# Changing all characters to lowercase
df['description'] = df['description'].apply(lambda x: x.lower())
df['title'] = df['title'].apply(lambda x: x.lower())

training_split = 0.8
train_size = int(training_split * len(df['description']))

train_data = df['description'][:train_size]
train_label = np.array(df['label'][:train_size])
validation_data = df['description'][train_size:]
validation_label = np.array(df['label'][train_size:])

train_list = train_data.to_numpy().flatten()
#print(train_list)

text_vec_layer = tf.keras.layers.TextVectorization()
text_vec_layer.adapt(train_list)
n_token = text_vec_layer.vocabulary_size()

epochs = 10

model = tf.keras.Sequential([
    text_vec_layer,
    tf.keras.layers.Embedding(input_dim=n_token,output_dim=64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(16,activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(4,activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
history = model.fit(train_data,train_label,epochs=epochs,validation_data=(validation_data,validation_label))

epochs_l = [*range(epochs)]

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs_l, acc, 'r')
plt.plot(epochs_l, val_acc, 'b')
plt.title('Training and validation accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["Accuracy", "Validation Accuracy"])
plt.show()