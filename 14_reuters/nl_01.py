import os
from bs4 import BeautifulSoup
from string import punctuation
from functools import reduce
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

punctuations = list(punctuation)

#Download and extracting text for file
#os.system('wget https://archive.ics.uci.edu/static/public/137/reuters+21578+text+categorization+collection.zip /kaggle/working/')
#os.mkdir('/kaggle/working/data/')
#zip_ref = zipfile.ZipFile('/kaggle/working/reuters+21578+text+categorization+collection.zip','r')
#zip_ref.extractall('/kaggle/working/data/')
#zip_ref.close()

#tar_ref = tarfile.open('/kaggle/working/data/reuters21578.tar.gz','r:gz')
#tar_ref.extractall('/kaggle/working/data/')

#Parsing text
cwd = os.getcwd()
path = cwd + '\\data\\reut2-000.sgm'
f = open(path,mode='r',encoding='utf8')
f = f.read()

soup = BeautifulSoup(f,'html.parser')
s = soup.find_all('body')

#Text preprocessing
s = list(map(lambda x:str(x),s))
s = list(map(lambda x:x.replace('<body>',''),s)) #Remove <body> tags
s = list(map(lambda x:x.replace('</body>',''),s)) #Remove </body> tags
s = list(map(lambda x:x.replace('\n',''),s)) #Remove newline tags
s = list(map(lambda x:x.replace('  ',''),s)) #Remove multiple spacing
s = list(map(lambda x:x.replace(',',''),s)) #Remove commas
s = list(map(lambda x:x.split('.'),s)) #Splitting a paragraph into multiple lines
s = reduce(lambda x,y:x+y,s) #Reducing the list of lists into lists

tokenizer = Tokenizer()
input_sequences = []
max_len = 0
tokenizer.fit_on_texts(s)

for line in s:
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