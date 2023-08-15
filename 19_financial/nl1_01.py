from datasets import load_dataset,Features,Value
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
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

pd.set_option('display.max_colwidth', None)
punctuations = list(punctuation)

wd = os.getcwd()
checkpoint_filepath = wd + '\\tmp\\checkpoint\\model.ckpt'
dataset = load_dataset('financial_phrasebank','sentences_75agree',split='train')

df = dataset.to_pandas()


df['sentence'] = df['sentence'].apply(lambda x: ''.join(i for i in x if not i in punctuations))
df['sentence'] = df['sentence'].apply(lambda x: x.lower())

print(df['label'].value_counts())
no_of_classes  = int(len(df['label'].value_counts().keys()))

#Hyperparameters
training_split = 0.8
maxlen = 2000
epochs = 30
train_size = int(training_split * len(df))

train_set, validation_set = train_test_split(df,test_size=1-training_split,stratify=df['label'])

train_data = train_set['sentence']
train_label = np.array(train_set['label'])
validation_data = validation_set['sentence']
validation_label = np.array(validation_set['label'])

#Oversampling to compensate for class imbalance

train_data = np.array(train_data).reshape(-1,1)
#sampler = RandomOverSampler()
sampler = RandomOverSampler(sampling_strategy={0:2146,1:1800,2:1600})
train_data, train_label = sampler.fit_resample(train_data,train_label)
train_data = train_data.flatten().squeeze()

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
    tf.keras.layers.Embedding(VOCAB_SIZE+1,128,input_length=maxlen),
    tf.keras.layers.Conv1D(128,3,activation='relu'),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(32,activation='relu'),      
    tf.keras.layers.Dense(16,activation='relu'),
    tf.keras.layers.Dense(no_of_classes,activation='softmax')
  ])

  return model

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
  
model = create_uncompiled_model()
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              metrics=['accuracy'])
history = model.fit(train_pad_trunc_seq, train_label, epochs=epochs,callbacks=[model_checkpoint_callback],validation_data=(validation_pad_trunc_seq,validation_label))

#This line is to expedite training (& epochs = 1). Comment out for full training. 
#model.load_weights(checkpoint_filepath) 

#yhat_valid = model.predict(validation_pad_trunc_seq)
#yhat_valid = np.argmax(yhat_valid,axis=1)
#yhat_list = list(yhat_valid)