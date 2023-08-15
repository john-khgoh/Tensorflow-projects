from os import getcwd, listdir
import numpy as np
#import pickle

from tensorflow.keras.preprocessing.text import Tokenizer

wd = getcwd()
path = wd + '\\data\\'
tokenizer = Tokenizer()

file_list = listdir(path)
file_list_len = len(file_list)
batch_size = 64
no_of_batches = 10 #Iterate through a subset of files in the directory
#no_of_batches = int(np.ceil(file_list_len/batch_size)) #Iterate through every file in the directory

cnt = 0
token_list = []
for i in range(0,no_of_batches):
    lower_limit = cnt * batch_size
    upper_limit = (cnt + 1) * batch_size
    temp_list = file_list[lower_limit:upper_limit]
    cnt += 1
    for j in range(batch_size):
        temp_file = open(path + temp_list[j],mode='r',encoding='utf8')
        temp_file = temp_file.read()
        temp_file = temp_file.lower().split('\n')
        tokenizer.fit_on_texts(temp_file)
        
total_words = len(tokenizer.word_index) + 1        
print(total_words)