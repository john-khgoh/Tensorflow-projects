from os import listdir,getcwd
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

pd.set_option('display.max_columns', 100)

def plot_series(time, series,label="", format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format,label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(False)

def to_datetime(time):
    time = time.split('-')
    year,month = time[0],time[1]
    date = dt(int(year),int(month),1)
    return date
    
wd = getcwd()
file = wd + '\\data-society-global-climate-change-data\\data\\globaltemperatures.csv'
checkpoint_filepath = wd + '\\checkpoints\\'
df = pd.read_csv(file)

time = list(map(lambda x:to_datetime(x),df['dt']))
#time_df = pd.DataFrame({'datetime':time})

series = df['landaveragetemperature'].to_numpy()

#Hyperparameters
split_time_ratio = 0.80 #The ratio between training and validation data
split_time = int(split_time_ratio * len(time)) #The cutoff point between training and validation data
shuffle_buffer = 128 #Doesn't seem very important. Just ensure it's a large number
batch_size = 64 #Just the ordinary neural network batch size i.e. how many training examples to train at once 
window_size = 12 #Window size should approximate the periodicity of the waveform e.g. if it repeats every 7 days, the window should be 7
epochs = 20
learning_rate = 0.001

conv_layer_filters = 64
lstm_layer = 12
dense_layer = [24,12,1]

series_valid_ori = series[split_time:]
time_valid_ori = time[split_time:]

plt.figure(figsize=(10, 6))
plot_series(time_valid_ori, series_valid_ori, label='Series_valid_ori')

#Normalization
mean = 9
#std = np.std(series)

pipe = make_pipeline(
    SimpleImputer(strategy='mean'),
    MinMaxScaler()
)
series = pipe.fit_transform(series.reshape(-1,1)).flatten()

time_train = time[:split_time]
time_valid = time[split_time:]
series_train = series[:split_time]
series_valid = series[split_time:]

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

#Rescale the values
#series_valid = rescale(series_valid,mean,std,signs_list_valid)
#rnn_forecast = rescale(rnn_forecast,mean,std,signs_list_valid)
#scaler = MinMaxScaler().fit(lax_series['LAX_Delayed'].to_numpy().reshape(-1,1))
#scaler.inverse_transform(rnn_forecast)

pipe2 = make_pipeline(
    SimpleImputer(strategy='mean',add_indicator=True),
    MinMaxScaler()
)
s = pipe2.fit(series_valid_ori.reshape(-1,1))
rnn_forecast = s.inverse_transform(rnn_forecast.reshape(-1,1)).flatten()

naive_forecast = np.array([mean] * len(rnn_forecast))
mae_naive = tf.keras.metrics.mean_absolute_error(series_valid, naive_forecast).numpy()
mae_rnn = tf.keras.metrics.mean_absolute_error(series_valid, rnn_forecast).numpy()

#for i in range(len(series_valid)):
#    print(series_valid[i],rnn_forecast[i])

#print(type(series_valid[0]),type(naive_forecast[0]),type(rnn_forecast[0]))
#print(series_valid,naive_forecast,rnn_forecast)
print(mae_naive," ",mae_rnn)

#plt.figure(figsize=(10, 6))
#plot_series(time_valid, series_valid)
plot_series(time_valid, rnn_forecast, label='rnn_forecast')
plot_series(time_valid, naive_forecast, label='naive_forecast')
plt.legend()
plt.show()
