#Multivariate with tf.keras.utils.timeseries_dataset_from_array()

from os import listdir,getcwd
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Input, Model
from sklearn.preprocessing import MinMaxScaler

class StandardScaler:
    def __init__(self,*args,**kwargs):
        self.mean = [] #Series means
        self.std = [] #Series standard deviations
        self.signs = None
    
    def fit(self,series):
        self.series = series
        
        #If series is a list
        if(isinstance(self.series,list)):
            self.mean = np.mean(self.series)
            self.std = np.std(self.series)
            
        #If series is an np.array
        elif(isinstance(self.series,np.ndarray)):
            #Check that it's 2D at most
            dim = self.series.ndim
            print(np.shape(self.series))
            print('Dimension: %d' %dim)
            if(dim>2):
                raise Exception('Expected 1 or 2 dimensional np.ndarray')
            else:
                #1D np.array
                if(dim==1):
                    self.mean = np.mean(self.series)
                    self.std = np.std(self.series)
                #2D np.array
                else:
                    #Iterate through columns and get column mean and std
                    r,c = np.shape(series)
                    for col in range(c):
                        self.mean.append(np.mean(series[:,col]))
                        self.std.append(np.std(series[:,col]))
        
        #If series is a pd.DataFrame or pd.Series
        elif(isinstance(self.series,(pd.core.frame.DataFrame,pd.core.series.Series))):
            self.mean = list(self.series.mean())
            self.std = list(self.series.std())
        
        #If series is not the above types
        else:
            raise Exception('Series must be of type list, numpy.ndarray, pandas.Series or pandas.DataFrame')
        
        print('Mean and Standard Deviation: %s,%s' %(self.mean,self.std))
    
    def fit_transform(self,series):
        self.series = series
        
        #If series is a list
        if(isinstance(self.series,list)):
            for i in self.series:
                diff = i - self.mean
                #positive values
                if(diff>0):
                    self.signs.append(1)
                #negative values
                else:
                    self.signs.append(-1)
                normal_val = abs(diff)/(self.std + np.float64(1e-12))
                #normal_val = diff/self.std
                normal_series.append(normal_val)
            return normal_series 
            
        #If series is an np.array
        elif(isinstance(self.series,np.ndarray)):
            #A very small value is added to the denominator to avoid zero division
            div = np.absolute(self.series) + np.float64(1e-12)
            self.signs = self.series/div
            
            results = np.absolute(self.series-self.mean)/(self.std + np.float64(1e-12))
            return results
        
        #If series is a pd.DataFrame or pd.Series 
        elif(isinstance(self.series,(pd.core.frame.DataFrame,pd.core.series.Series))):
            pass
            
    def inverse_transform(self,series):
        self.series = series
        rescaled_list = []
        for i,j in enumerate(self.series):
            #positive values
            diff = (j * self.std)
            if(self.signs_list[i]>0):
                val = diff + self.mean
            #negative values
            else:
                val = self.mean - diff
            rescaled_list.append(val)
        return rescaled_list

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
series_valid_copy = series[split_time:]
no_of_cols = series.shape[1]

a = [[1,2,3,4],[4,5,6,7]]
#a = [[1,2,3,4],[2,4,6,8],[3,5,7,9]]
#a = [[[1,2,3],[1,2,3]],[[1,2,3],[1,2,3]]]
#a = np.random.rand(10,10) * 1000
#a = series
#a = np.array(a)
#print(np.mean(a))
#print(a)
print(type(a))
s = StandardScaler()
s.fit(a)
a = s.fit_transform(a)
#print(a)