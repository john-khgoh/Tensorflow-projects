import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.models import Sequential
#from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from string import punctuation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#pd.options.display.max_colwidth = 2000

def limit_str_size(str,word_limit=500):
    str_list = str.split(' ')
    str_list = str_list[:word_limit]
    return str_list
    
punctuations = list(punctuation)
word_limit = 100
doc_limit = 500

#Getting the dataset
ds, info = tfds.load('imdb_reviews', split='train[:100%]', with_info=True)

no_of_classes = info.features['label'].num_classes
#print(no_of_classes)

# Convert the TFDS dataset to pandas dataframe for preprocessing
df = tfds.as_dataframe(ds)
df = df[:doc_limit]
df['text'] = df['text'].apply(lambda x: x.decode('utf-8'))

# Removing special characters
df['text'] = df['text'].apply(lambda x: ''.join(i for i in x if not i in punctuations))

# Changing all characters to lowercase
df['text'] = df['text'].apply(lambda x: x.lower())

df['text'] = df['text'].apply(lambda x: limit_str_size(x,word_limit))

#df_new = df[['text']].copy()

tokenizer = Tokenizer()

input_sequences = []
max_len = 0

tokenizer.fit_on_texts(df['text'])
for line in df['text']:
    token_list = []
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1,len(token_list)):
        n_gram_sequence = token_list[:i+1]
        len_n_gram_sequence = len(n_gram_sequence)
        if(len_n_gram_sequence>max_len):
            max_len = len_n_gram_sequence
        input_sequences.append(n_gram_sequence)

total_words = len(tokenizer.word_index) + 1
input_sequences = pad_sequences(input_sequences,maxlen=max_len,padding='pre')
xs, labels = input_sequences[:,:-1],input_sequences[:,-1]
ys = to_categorical(labels,num_classes=total_words)

#Hyperparameters
embedding_dim = 100
lstm_units_1 = 150
learning_rate = 0.01
epochs = 30

model = Sequential([
    Embedding(total_words,embedding_dim,input_length=max_len-1),
    #Bidirectional(LSTM(lstm_units_1, return_sequences=True)),
    Bidirectional(LSTM(lstm_units_1)),
    Dense(total_words,activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate),
    metrics = ['accuracy']
)

model.summary()
history = model.fit(xs,ys,epochs=epochs)

#Model seed and next words
seed_text = 'Today I saw a'
next_words = 100

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list],maxlen=max_len-1,padding='pre')
    prob = model.predict(token_list,verbose=0)
    predicted = np.argmax(prob,axis=-1)[0]
    if predicted != 0:
        output_word = tokenizer.index_word[predicted]
        seed_text += ' ' + output_word

print(seed_text)

