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

tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(train_data)
word_index = tokenizer.word_index
VOCAB_SIZE = len(word_index)

train_sequences = tokenizer.texts_to_sequences(train_data)
train_pad_trunc_seq = pad_sequences(train_sequences, truncating='post', maxlen=50)

validation_sequences = tokenizer.texts_to_sequences(validation_data)
validation_pad_trunc_seq = pad_sequences(validation_sequences, truncating='post', maxlen=50)

'''
def create_uncompiled_model():
  model = tf.keras.Sequential([
      tf.keras.layers.Embedding(VOCAB_SIZE+1,64,input_length=50),
      tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
      tf.keras.layers.Dense(16,activation='relu'),
      tf.keras.layers.Dropout(0.25),
      tf.keras.layers.Dense(4,activation='softmax')
  ])
  return model

def adjust_learning_rate():
  model = create_uncompiled_model()
  lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-6 * 20**(epoch / 20))
  optimizer = tf.keras.optimizers.Adam(learning_rate=1e-8)
  model.compile(loss='sparse_categorical_crossentropy',
                optimizer=optimizer, 
                metrics=["accuracy"]) 
  history = model.fit(train_pad_trunc_seq,train_label,epochs=50,validation_data=(validation_pad_trunc_seq,validation_label),callbacks=[lr_schedule])

  return history
'''
epochs = 10
model = create_uncompiled_model()
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              metrics=['accuracy'])
history = model.fit(train_pad_trunc_seq, train_label, epochs=epochs,
                    validation_data=(validation_pad_trunc_seq, validation_label))
'''
lr_history = adjust_learning_rate()
plt.semilogx(lr_history.history["lr"], lr_history.history["loss"])
plt.axis([1e-6, 1, 0, 3])
'''
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