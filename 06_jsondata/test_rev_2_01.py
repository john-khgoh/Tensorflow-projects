#Exercise to backfill past data given the present time series data, as opposed to forecasting

from os import listdir,getcwd
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime as dt
from sklearn.preprocessing import MinMaxScaler

def plot_series(time, series,label="", format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format,label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(False)

def to_datetime(time):
    time = time.split('/')
    year,month = time[0],time[1]
    date = dt(int(year),int(month),1)
    return date
    
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

lax_series = pd.DataFrame({'LAX_Delayed':lax_delayed,'LAX_Cancelled':lax_cancelled,'LAX_Ontime':lax_on_time,'LAX_Diverted':lax_diverted,'LAX_Total':lax_total})
lax_series_bkp = lax_series
sfo_series = pd.DataFrame({'SFO_Delayed':sfo_delayed,'SFO_Cancelled':sfo_cancelled,'SFO_Ontime':sfo_on_time,'SFO_Diverted':sfo_diverted,'SFO_Total':sfo_total})
#time = np.arange(len(lax_series),dtype="int32")
series = lax_series['LAX_Delayed'].to_numpy()

time = list(map(lambda x:to_datetime(x),lax_time))
time_df = pd.DataFrame({'datetime':time})
#print(time_df)

#Hyperparameters
split_time_ratio = 0.80 #The ratio between training and validation data
split_time = int(split_time_ratio * len(time)) #The cutoff point between training and validation data
shuffle_buffer = 128 #Doesn't seem very important. Just ensure it's a large number
batch_size = 64 #Just the ordinary neural network batch size i.e. how many training examples to train at once 
window_size = 7 #Window size should approximate the periodicity of the waveform e.g. if it repeats every 7 days, the window should be 7
epochs = 30
learning_rate = 0.01

conv_layer_filters = 64
lstm_layer = 12
dense_layer = [24,12,1]

series_valid_ori = series[:-split_time]
time_valid_ori = time[:-split_time]

plt.figure(figsize=(10, 6))
plot_series(time_valid_ori, series_valid_ori, label='Series_valid_ori')

#Normalization
series_mean = np.mean(series)
series_std = np.std(series)
series = abs(series - series_mean)/series_std
#scaler = MinMaxScaler().fit(series.reshape(-1,1))
#series = scaler.transform(series.reshape(-1,1)).flatten()
#print(series_mean)

#To backfill, we have to move the colon to the other side and add a negative sign 
#This is the most important difference for backfilling as opposed to forecasting
time_train = time[-split_time:]
time_valid = time[:-split_time]
series_train = series[-split_time:]
series_valid = series[:-split_time]

#series_train = abs(series_train - series_mean)/series_std
#series_valid = abs(series_valid - series_mean)/series_std

#When then flip the data to reverse the order of the time series data
time_train = time_train
time_valid = time_valid
series_train = np.flip(series_train)
series_valid = np.flip(series_valid)

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

#Windowing of the dataset
#def window(series,window_size,shuffle_buffer,batch_size):
#    dataset = tf.data.Dataset.from_tensor_slices(series)
#    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
#    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
#    dataset = dataset.shuffle(shuffle_buffer)
#    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
#    dataset = dataset.batch(batch_size).prefetch(1)
#    return dataset
#dataset = window(series_train,window_size,shuffle_buffer,batch_size)

model = tf.keras.models.Sequential([ 
        tf.keras.layers.Conv1D(filters=conv_layer_filters,kernel_size=5,strides=1,padding='causal',activation='relu',input_shape=[None,1]),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_layer,return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_layer)),
        tf.keras.layers.Dense(dense_layer[0],activation='relu'),
        tf.keras.layers.Dense(dense_layer[1],activation='relu'),
        tf.keras.layers.Dense(dense_layer[2],activation='relu')
    ]) 
    
model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=["mae"])  
                
history = model.fit(train_ds,epochs=epochs)

def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(batch_size).prefetch(1)
    forecast = model.predict(ds)
    return forecast
    
forecast_series = series[split_time - window_size:-1]
rnn_forecast = model_forecast(model, forecast_series, window_size).flatten()

def rescale(series,mean,std):
    series = (series * std) + mean
    return series

#Rescale the values
#series_valid = rescale(series_valid,series_mean,series_std)
rnn_forecast = rescale(rnn_forecast,series_mean,series_std)
#new_scaler = MinMaxScaler().fit(lax_series_bkp['LAX_Delayed'].to_numpy().reshape(-1,1))
#rnn_forecast = new_scaler.inverse_transform(np.array(rnn_forecast).reshape(-1,1)).flatten()
#series_valid = new_scaler.inverse_transform(np.array(series_valid).reshape(-1,1)).flatten()

#Flipping the results back
rnn_forecast = np.flip(rnn_forecast)
series_valid = np.flip(series_valid)

naive_forecast = np.array([series_mean] * len(rnn_forecast))
mae_naive = tf.keras.metrics.mean_absolute_error(series_valid, naive_forecast).numpy()
mae_rnn = tf.keras.metrics.mean_absolute_error(series_valid, rnn_forecast).numpy()

#print(type(series_valid[0]),type(naive_forecast[0]),type(rnn_forecast[0]))
#print(series_valid,naive_forecast,rnn_forecast)
print(mae_naive," ",mae_rnn)

#plt.figure(figsize=(10, 6))
#plot_series(time_valid, series_valid, label='series_valid')
plot_series(time_valid, rnn_forecast, label='rnn_forecast')
plot_series(time_valid, naive_forecast, label='naive_forecast')
plt.legend()
plt.show()
