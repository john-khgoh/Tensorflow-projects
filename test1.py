#import tensorflow as tf
import numpy as np
from os import getcwd
import pandas as pd
from time import sleep
import re

pd.set_option('display.max_columns', 100)
#pd.set_option('display.max_rows', 10000)

wd = getcwd()
train_file = wd + "\\train.csv"
test_file = wd + "\\test.csv"
output_file = wd + "\\output.csv"

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

#Function that takes a list of dataframes and concatenates them into one long dataframe with category labels
def df_train_test_concatenator(list_of_df):
    from functools import reduce
    cat_list = []
    cnt = 1
    for df in list_of_df:
        cat_list.append(len(df) * [cnt])
        cnt += 1
    cat_list = reduce(lambda x,y:x+y,cat_list)
    cat_df = pd.DataFrame(cat_list,columns=['Category'])
    df = pd.concat(list_of_df,axis=0)
    df = df.reset_index(drop=True)
    df = pd.concat([df,cat_df],axis=1)
    return df

#Function that takes a dataframe and separates them into numerical, categorical and unknown dataframes
def numerical_categorical_separator(df):
    df_list = []
    numerical_df = pd.DataFrame()
    categorical_df = pd.DataFrame()
    unknown_df = pd.DataFrame()
    list_of_cols = df.columns
    for i,j in enumerate(df.dtypes):
        if(j=='int64'):
            numerical_df = pd.concat([numerical_df,df[list_of_cols[i]]],axis=1)
        elif(j=='float64'):
            numerical_df = pd.concat([numerical_df,df[list_of_cols[i]]],axis=1)
        elif(j=='object'):
            categorical_df = pd.concat([categorical_df,df[list_of_cols[i]]],axis=1)
        else:
            unknown_df = pd.concat([unknown_df,df[list_of_cols[i]]],axis=1)
    df_list.append(numerical_df)
    df_list.append(categorical_df)
    df_list.append(unknown_df)
    return df_list

#Function that searches for a substring in a list of strings e.g. searching for a pattern among variable names
def find_substring_from_list(substring,list):
    match_list = []
    for i in list:
        result = re.search(substring,i,re.IGNORECASE) #Ignore case
        if(result is not None):
            match_list.append(result.string)
    return match_list
    
comb_df = df_train_test_concatenator([train_df,test_df])

df_list = numerical_categorical_separator(comb_df)
numerical_df,categorical_df = df_list[0],df_list[1]
numerical_df_cols_list = list(numerical_df.columns)

#Finding the list of columns like 'year,yr or mo' because these are actually categorical columns that look like numerical columns
categorical_number_list = find_substring_from_list('year',numerical_df_cols_list)
categorical_number_list += find_substring_from_list('yr',numerical_df_cols_list)
categorical_number_list += find_substring_from_list('mo',numerical_df_cols_list)
categorical_number_list = list(set(categorical_number_list))

categorical_df_2 = numerical_df.filter(categorical_number_list)
categorical_df = pd.concat([categorical_df,categorical_df_2],axis=1)
numerical_df = numerical_df.drop(columns=categorical_number_list)

print(numerical_df)