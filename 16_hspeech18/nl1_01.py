from datasets import load_dataset
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from string import punctuation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

punctuations = list(punctuation)

wd = os.getcwd()
checkpoint_filepath = wd + '\\tmp\\checkpoint\\model.ckpt'
dataset = load_dataset('hate_speech18',split='train')
df = dataset.to_pandas()

df['text'] = df['text'].apply(lambda x: ''.join(i for i in x if not i in punctuations))
df['text'] = df['text'].apply(lambda x: x.lower())

#Checking number of classes and data distribution for label imbalance
#print(df['label'].value_counts())

#Hyperparameters
training_split = 0.8
maxlen = 1200
epochs = 5
train_size = int(training_split * len(df['text']))

train_set, validation_set = train_test_split(df,test_size=1-training_split,stratify=df['label'])
print(train_set['label'].value_counts())
print(validation_set['label'].value_counts())

train_data = train_set['text']
train_label = np.array(train_set['label'])
validation_data = validation_set['text']
validation_label = np.array(validation_set['label'])

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

def create_uncompiled_model():
  
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE+1,64,input_length=maxlen),
    tf.keras.layers.Conv1D(64,5,activation='relu'),
    tf.keras.layers.GlobalAveragePooling1D(),
    #tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(32,activation='relu'),
    tf.keras.layers.Dense(16,activation='relu'),
    tf.keras.layers.Dense(4,activation='softmax')
  ])

  return model

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='accuracy',
    mode='max',
    save_best_only=True)
  
model = create_uncompiled_model()
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-4),
              metrics=['accuracy'])            
history = model.fit(train_pad_trunc_seq, train_label, epochs=epochs,callbacks=[model_checkpoint_callback])

#This line is to expedite training (& epochs = 1). Comment out for full training. 
#model.load_weights(checkpoint_filepath) 

yhat_valid = model.predict(validation_pad_trunc_seq)
yhat_valid = np.argmax(yhat_valid,axis=1)
yhat_list = list(yhat_valid)
print(yhat_list)

#yhat_valid_df = pd.DataFrame(yhat_valid)
#print(yhat_valid_df)
#cols = yhat_valid_df.columns