#Multivariate with tf.keras.utils.timeseries_dataset_from_array()

from os import listdir,getcwd
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#from tensorflow.keras import Input, Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

class StandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self,*args,**kwargs):
        self.mean = [] #Series means
        self.std = [] #Series standard deviations
        self.signs = None
    
    def fit(self,series):
        self.series = np.array(series)
        
        #Check that it's 2D at most
        dim = self.series.ndim
        #print(np.shape(self.series))
        #print('Dimension: %d' %dim)
        if(dim>2):
            raise Exception('Expected 1 or 2 dimensional series (list, numpy.array or pandas.dataframe). Received %d dimensional series.' %(dim))
        else:
            #1D np.array
            if(dim==1):
                self.mean = np.mean(self.series)
                self.std = np.std(self.series)
            #2D np.array
            else:
                #Iterate through columns and get column mean and std
                r,c = np.shape(self.series)
                for col in range(c):
                    self.mean.append(np.mean(self.series[:,col]))
                    self.std.append(np.std(self.series[:,col]))
                    
        return self
        #print('Mean and Standard Deviation: %s,%s' %(self.mean,self.std))
    
    def transform(self,series):
        self.series = np.array(series)
        
        div = np.absolute(self.series) + np.float64(1e-12)
        
        results = self.series-self.mean
        abs_results = abs(results) + np.float64(1e-12)
        self.signs = results/abs_results
        #print(self.signs)
        
        results = abs_results/(self.std + np.float64(1e-12))
        #results = abs_results/self.std
        return results
          
    def inverse_transform(self,series):
        self.series = np.array(series)
        
        results = (self.series * self.std * self.signs) + self.mean
        return results
       
        
wd = getcwd()
time = []

#Lists for LAX
lax_time = []
lax_delayed = []
lax_cancelled = []
lax_on_time = []
lax_diverted = []
lax_total = []

#List for SFO
sfo_time = []
sfo_delayed = []
sfo_cancelled = []
sfo_on_time = []
sfo_diverted = []
sfo_total = []

with open('airlines.json') as file:
    file_content = file.read()
    
parsed_json = json.loads(file_content)

#Only print out data for LAX and SFO
for i in parsed_json:  
    if(i['Airport']['Code']=='LAX'):
        lax_time.append(i['Time']['Label'])
        lax_delayed.append(i['Statistics']['Flights']['Delayed'])
        lax_cancelled.append(i['Statistics']['Flights']['Cancelled'])
        lax_on_time.append(i['Statistics']['Flights']['On Time'])
        lax_diverted.append(i['Statistics']['Flights']['Diverted'])
        lax_total.append(i['Statistics']['Flights']['Total'])
    elif(i['Airport']['Code']=='SFO'):
        sfo_time.append(i['Time']['Label'])
        sfo_delayed.append(i['Statistics']['Flights']['Delayed'])
        sfo_cancelled.append(i['Statistics']['Flights']['Cancelled'])
        sfo_on_time.append(i['Statistics']['Flights']['On Time'])
        sfo_diverted.append(i['Statistics']['Flights']['Diverted'])
        sfo_total.append(i['Statistics']['Flights']['Total'])

lax_df = pd.DataFrame({'LAX_Delayed':lax_delayed,'LAX_Cancelled':lax_cancelled,'LAX_Ontime':lax_on_time,'LAX_Diverted':lax_diverted,'LAX_Total':lax_total})
sfo_df = pd.DataFrame({'SFO_Delayed':sfo_delayed,'SFO_Cancelled':sfo_cancelled,'SFO_Ontime':sfo_on_time,'SFO_Diverted':sfo_diverted,'SFO_Total':sfo_total})
time = np.arange(len(lax_df),dtype="int32")

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(False)

#Hyperparameters
split_time_ratio = 0.80 #The ratio between training and validation data
split_time = int(split_time_ratio * len(time)) #The cutoff point between training and validation data
shuffle_buffer = 128 #Doesn't seem very important. Just ensure it's a large number. 
batch_size = 64 #Just the ordinary neural network batch size i.e. how many training examples to train at once 
window_size = 6 #Window size should approximate the periodicity of the waveform e.g. if it repeats every 7 days, the window should be 7
epochs = 40
learning_rate = 0.02

conv_layer_filters = 32
lstm_layer = 12
dense_layer = [24,12,1]

#Keep the columns of interest
series = lax_df[['LAX_Delayed','LAX_Cancelled']]
#series = lax_df[['LAX_Delayed']]
#series_valid_copy = series[split_time:]
#no_of_cols = series.shape[1]

#a = [0,0,0,0,0]
#a = [[1,1,1,1],[2,2,2,2]]
#a = [[1,2,-3,0],[-2,4,6,0],[3,5,7,0]]
#a = [[[1,2,3],[1,2,3]],[[1,2,3],[1,2,3]]]
#a = np.random.rand(10,10) * 1000
a = series
#a = np.array(a)
#print(np.mean(a))
#print(a)
#print(type(a))
#s = StandardScaler()
#s.fit(a)
#a = s.fit_transform(a)
#print(a)
#a = s.inverse_transform(a)
#print(a)

test_pipeline = make_pipeline(
    StandardScaler()
    #MinMaxScaler()
)
a = test_pipeline.fit_transform(a)
a = test_pipeline.inverse_transform(a)
print(a)