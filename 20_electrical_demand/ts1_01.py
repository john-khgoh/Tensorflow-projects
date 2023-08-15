#Univariate time series

from os import getcwd, listdir
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime as dt
from keras.layers import Dense, Flatten, LSTM, Activation
from keras.layers import Dropout, RepeatVector, TimeDistributed
from keras import Input, Model
from sklearn.base import BaseEstimator, TransformerMixin
#from sklearn.preprocessing import MinMaxScaler, StandardScaler
from datasets import load_dataset

#pd.set_option('display.max_rows', 10000)

wd = getcwd()
checkpoint_filepath = wd + '\\checkpoint\\model.ckpt'

dataset = load_dataset('rajistics/electricity_demand',split='train')

df = dataset.to_pandas()
df.rename(columns={'__index_level_0__':'time'},inplace=True)

def plot_series(time, series,label="", format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format,label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(False)

class UnStandardScaler(BaseEstimator,TransformerMixin):
    def __init__(self,*args,**kwargs):
        self.mean = [] #Series means
        self.std = [] #Series standard deviations
        self.signs = None
        
    def fit(self,series):
        self.series = series
        
        #If series is a list
        if(isinstance(self.series,list)):
            self.mean = np.nanmean(self.series)
            self.std = np.nanstd(self.series)
            
        #If series is an np.array
        elif(isinstance(self.series,np.ndarray)):
            #Check that it's 2D at most
            dim = self.series.ndim
            if(dim>2):
                raise Exception('Expected 1 or 2 dimensional np.ndarray')
            else:
                #1D np.array
                if(dim==1):
                    self.mean = np.nanmean(self.series)
                    self.std = np.nanstd(self.series)
                #2D np.array
                else:
                    #Iterate through columns and get column mean and std
                    r,c = np.shape(series)
                    for row in range(r):
                        self.mean.append(np.nanmean(series[row,:]))
                        self.std.append(np.nanstd(series[row,:]))
                    self.mean = np.expand_dims(np.array(self.mean).transpose(),1)
                    self.std = np.expand_dims(np.array(self.std).transpose(),1)
                    
        #If series is a pd.DataFrame or pd.Series
        elif(isinstance(self.series,(pd.core.frame.DataFrame,pd.core.series.Series))):
            self.mean = list(self.series.mean())
            self.std = list(self.series.std())
        
        self.series = np.array(series)
        div = np.absolute(self.series) + np.float64(1e-12)
        results = self.series-self.mean
        abs_results = abs(results) + np.float64(1e-12)
        self.signs = results/abs_results
        
        return self
    
    def transform(self,series):
        self.series = np.array(series)
        
        div = np.absolute(self.series) + np.float64(1e-12)
        
        #Factorized implementation
        results = self.series-self.mean
        abs_results = abs(results) + np.float64(1e-12)
        self.signs = results/abs_results
        
        results = abs_results/(self.std + np.float64(1e-12))
        return results
          
    def inverse_transform(self,series):
        self.series = np.array(series)
        
        #Factorized implementation
        results = (self.series * self.std * self.signs) + self.mean
        return results
        
def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(batch_size).prefetch(1)
    forecast = model.predict(ds)
    return forecast

def build_model():
    model1_inputs = Input(shape=(window_size,input_dims,))
    model1_outputs = Input(shape=(output_dims,))

    net1 = LSTM(lstm_layer, return_sequences=True)(model1_inputs)
    net1 = LSTM(lstm_layer, return_sequences=True)(net1)
    net1 = LSTM(lstm_layer, return_sequences=False)(net1)
    net1 = Dense(output_dims, activation='relu')(net1)
    model1_outputs = net1

    model1 = Model(inputs=model1_inputs, outputs = model1_outputs, name='model1')

    model1.compile(loss='mae',
                      optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      metrics=["mae"])
    return model1
    
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_mae',
    mode='min',
    save_best_only=True)
    
time = np.arange(len(df),dtype="int32")    
split_time_ratio = 0.80 #The ratio between training and validation data
split_time = int(split_time_ratio * len(time))
shuffle_buffer = 128 #Doesn't seem very important. Just ensure it's a large number. 
batch_size = 64 #Just the ordinary neural network batch size i.e. how many training examples to train at once 
window_size = 12 #Window size should approximate the periodicity of the waveform e.g. if it repeats every 7 days, the window should be 7
epochs = 30
learning_rate = 8e-4
input_dims = 2
output_dims = 1 # number of classes
lstm_layer = 12

series = df[['Demand','Temperature']]
#print(series)

series_valid_bkp = series[split_time:]
time_valid_bkp = time[split_time:]

plt.figure(figsize=(10, 6))
plot_series(time_valid_bkp, series_valid_bkp['Demand'], label='Series_valid')

#Normalization
mean = np.mean(series_valid_bkp)
#std = np.std(series)

time_train = time[:split_time]
time_valid = time[split_time:]
series_train = series[:split_time]
series_valid = series[split_time:]

s_train = UnStandardScaler().fit(series_train)
series_train = s_train.transform(series_train)
s_valid = UnStandardScaler().fit(series_valid)
series_valid = s_valid.transform(series_valid)

train_ds = tf.keras.utils.timeseries_dataset_from_array(
        series_train,
        targets = series_train,
        sequence_length = window_size,
        batch_size = batch_size,
        shuffle = True
    )
valid_ds = tf.keras.utils.timeseries_dataset_from_array(
    series_valid,
    targets = series_valid,
    sequence_length = window_size,
    batch_size = batch_size,
    shuffle = True
)

model = build_model()
history = model.fit(train_ds,epochs=epochs,callbacks=[model_checkpoint_callback],validation_data=valid_ds)
#history = model.fit(train_ds,epochs=epochs,callbacks=[model_checkpoint_callback])
       
forecast_series = series[split_time - window_size:-1]
rnn_forecast = model_forecast(model, forecast_series, window_size)
#rnn_forecast = s_valid.inverse_transform(rnn_forecast)
rnn_forecast = rnn_forecast[:,:1]
series_valid_bkp_out = np.array(series_valid_bkp['Demand']).reshape(-1,1)
#print(np.shape(rnn_forecast),np.shape(series_valid_bkp_out))

#print(np.shape(series_valid_bkp_out),np.shape(rnn_forecast))
#s_output = StandardScaler().fit(series_valid_bkp_out)
#s_output = MinMaxScaler().fit(series_valid_bkp_out)
s_output = UnStandardScaler()
_ = s_output.fit(series_valid_bkp_out)
#series_valid_bkp_out = s_output.fit(series_valid_bkp_out)
rnn_forecast = s_output.inverse_transform(rnn_forecast)

naive_forecast = np.array([mean[0]] * len(rnn_forecast)).flatten()

#print(np.shape(naive_forecast),np.shape(series_valid_bkp['Demand']),np.shape(rnn_forecast))

mae_naive = tf.keras.metrics.mean_absolute_error(series_valid_bkp['Demand'], naive_forecast).numpy()
mae_rnn = tf.keras.metrics.mean_absolute_error(series_valid_bkp['Demand'], rnn_forecast.squeeze()).numpy()
#print(np.array(series_valid_bkp['Demand']),rnn_forecast)
print(mae_naive," ",mae_rnn)

plot_series(time_valid, rnn_forecast, label='rnn_forecast')
plot_series(time_valid, naive_forecast, label='naive_forecast')
plt.legend()
plt.show()
