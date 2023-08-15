from os import getcwd, listdir, system
from google.colab import drive
import numpy as np
from random import shuffle
import zipfile

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.models import Sequential
#from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

#drive.mount('/content/gdrive')

#wd = getcwd() + '/gdrive/MyDrive/data/'

# with zipfile.ZipFile(wd + 'archive.zip','r') as z:
#  z.extractall('/data/')

path = '/data/data/'
tokenizer = Tokenizer()

file_list = listdir(path)
shuffle(file_list)
file_list_limit = 50
file_list = file_list[:file_list_limit]

input_sequences = []
max_len = 0
for file in file_list:
    temp_file = open(path + file,mode='r',encoding='utf8')
    temp_file = temp_file.read()
    temp_file = temp_file.lower().split('\n') #equivalent of a corpus
    tokenizer.fit_on_texts(temp_file)
    for line in temp_file:
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
