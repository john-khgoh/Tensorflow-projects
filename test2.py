#import tensorflow as tf
from os import getcwd
import pandas as pd
import numpy as np
import re
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline

#pd.set_option('display.max_columns', 100)
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
categorical_cols =  categorical_df.columns
numerical_cols = numerical_df.columns

#Unique value by column and count of each value

for i in categorical_cols:
    #print(categorical_df[i].value_counts())
    break

numerical_cols = numerical_df.columns
for i in numerical_cols:
    #print(numerical_df[i].value_counts())
    break

#Distribution of NAs by column
cat_col_na_dict = {}
for i in categorical_cols:
    cat_col_na_dict[i] = categorical_df[i].isna().sum()

num_col_na_dict = {}
for i in numerical_cols:
    num_col_na_dict[i] = numerical_df[i].isna().sum()

#Sort the dictionary by value
sorted_num_col_na_dict = dict(sorted(num_col_na_dict.items(), key=lambda item: item[0]))
sorted_cat_col_na_dict = dict(sorted(cat_col_na_dict.items(), key=lambda item: item[0]))
#print(sorted_num_col_na_dict)
#print(sorted_cat_col_na_dict)

#Columns that cannot be imputed because NaN is part of the data description
legal_na_cols_list = ['Alley','MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PoolQC','Fence','MiscFeature']
categorical_df_3 = categorical_df.filter(legal_na_cols_list)
categorical_df = categorical_df.drop(columns=legal_na_cols_list) #The other NaNs in categorical_df can be imputed

#Pipeline:
#Caterorical_df_3: One-hot Encoder
#Categorical_df: Imputation -> One-hot encoder
#Numerical_df: Imputation -> Normalize

categorical_col_list = list(categorical_df.columns)
numerical_col_list = list(numerical_df.columns)
'''
cat_pipeline = make_pipeline(
    SimpleImputer(strategy='most_frequent'),
    OneHotEncoder(handle_unknown='ignore')
)

num_pipeline = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler()
)

cat3_pipeline = make_pipeline(
    OneHotEncoder(handle_unknown='ignore')
)

ct = ColumnTransformer([
    ("num",num_pipeline,numerical_col_list),
    ("cat",cat_pipeline,categorical_col_list),
    ("cat3",cat3_pipeline,legal_na_cols_list),
])

prepared_data = ct.fit_transform(train_df)
'''