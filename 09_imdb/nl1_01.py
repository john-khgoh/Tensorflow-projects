import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from string import punctuation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

punctuations = list(punctuation)

#Getting the dataset
ds, info = tfds.load('imdb_reviews', split='train[:100%]', with_info=True)

# Convert the TFDS dataset to pandas dataframe for preprocessing
df = tfds.as_dataframe(ds)
df['text'] = df['text'].apply(lambda x: x.decode('utf-8'))

# Removing special characters
df['text'] = df['text'].apply(lambda x: ''.join(i for i in x if not i in punctuations))

# Changing all characters to lowercase
df['text'] = df['text'].apply(lambda x: x.lower())

#Checking number of classes and data distribution for label imbalance
#print(df['label'].value_counts())

#Hyperparameters
training_split = 0.8
maxlen = 1200
epochs = 15
train_size = int(training_split * len(df['text']))

#Train and testing split
train_data = df['text'][:train_size]
train_label = np.array(df['label'][:train_size])
validation_data = df['text'][train_size:]
validation_label = np.array(df['label'][train_size:])

#Tokenization
tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(train_data)
word_index = tokenizer.word_index
VOCAB_SIZE = len(word_index)

#Turning text to sequences and padding
train_sequences = tokenizer.texts_to_sequences(train_data)
train_pad_trunc_seq = pad_sequences(train_sequences, truncating='post', maxlen=maxlen)

validation_sequences = tokenizer.texts_to_sequences(validation_data)
validation_pad_trunc_seq = pad_sequences(validation_sequences, truncating='post', maxlen=maxlen)

#Uncompiled model
def create_uncompiled_model():
  '''
  model = tf.keras.Sequential([
      tf.keras.layers.Embedding(VOCAB_SIZE+1,64,input_length=maxlen),
      tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
      tf.keras.layers.Dense(64,activation='relu'),
      #tf.keras.layers.Dropout(0.30),
      #tf.keras.layers.Dense(8,activation='relu'),
      #tf.keras.layers.Dropout(0.30),
      tf.keras.layers.Dense(1,activation='sigmoid')
  ])
  '''
  
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE+1,64,input_length=maxlen),
    tf.keras.layers.Conv1D(64,5,activation='relu'),
    tf.keras.layers.GlobalAveragePooling1D(),
    #tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(32,activation='relu'),
    tf.keras.layers.Dense(16,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
  ])

  return model
  
model = create_uncompiled_model()
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Nadam(learning_rate=8e-5),
              metrics=['accuracy'])
history = model.fit(train_pad_trunc_seq, train_label, epochs=epochs,
                    validation_data=(validation_pad_trunc_seq, validation_label))
                    
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