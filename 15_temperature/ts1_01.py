from os import getcwd, listdir
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from keras.layers import Dense, Flatten, LSTM, Activation
from keras.layers import Dropout, RepeatVector, TimeDistributed
from keras import Input, Model
from sklearn.base import BaseEstimator, TransformerMixin

wd = getcwd()
file = wd + '\\daily-min-temperatures.csv'
checkpoint_filepath = wd + '\\checkpoints\\'
df = pd.read_csv(file)

class StandardScaler(BaseEstimator,TransformerMixin):
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
                    for row in range(r):
                        self.mean.append(np.mean(series[row,:]))
                        self.std.append(np.std(series[row,:]))
                    self.mean = np.expand_dims(np.array(self.mean).transpose(),1)
                    self.std = np.expand_dims(np.array(self.std).transpose(),1)
                    
        #If series is a pd.DataFrame or pd.Series
        elif(isinstance(self.series,(pd.core.frame.DataFrame,pd.core.series.Series))):
            self.mean = list(self.series.mean())
            self.std = list(self.series.std())
        
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
        
def plot_series(time, series,label="", format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format,label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(False)
    
def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(batch_size).prefetch(1)
    forecast = model1.predict(ds)
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
epochs = 20
learning_rate = 1e-7
input_dims = 1
output_dims = 1 # number of classes
lstm_layer = 12

conv_layer_filters = 64
lstm_layer = 12
dense_layer = [24,12,1]

series = df['Temp']

series_valid_bkp = series[split_time:]
time_valid_bkp = time[split_time:]

plt.figure(figsize=(10, 6))
plot_series(time_valid_bkp, series_valid_bkp, label='Series_valid')

#Normalization
mean = np.mean(series)
std = np.std(series)

time_train = time[:split_time]
time_valid = time[split_time:]
series_train = series[:split_time]
series_valid = series[split_time:]

series_train = np.array(series_train).reshape(-1,1)
series_valid = np.array(series_valid).reshape(-1,1)

s_train = StandardScaler().fit(series_train)
series_train = s_train.transform(series_train)
s_valid = StandardScaler().fit(series_valid)
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

model1 = build_model()
history = model1.fit(train_ds,epochs=epochs,callbacks=[model_checkpoint_callback],validation_data=valid_ds)
       
forecast_series = series[split_time - window_size:-1]
rnn_forecast = model_forecast(model1, forecast_series, window_size)

rnn_forecast = s_valid.inverse_transform(rnn_forecast).flatten()

#print(series_valid_bkp)
#print(rnn_forecast)

naive_forecast = np.array([mean] * len(rnn_forecast)).flatten()
print(len(naive_forecast),len(series_valid_bkp),len(rnn_forecast))
mae_naive = tf.keras.metrics.mean_absolute_error(series_valid_bkp, naive_forecast).numpy()
mae_rnn = tf.keras.metrics.mean_absolute_error(series_valid_bkp, rnn_forecast).numpy()
print(mae_naive," ",mae_rnn)

plot_series(time_valid, rnn_forecast, label='rnn_forecast')
plot_series(time_valid, naive_forecast, label='naive_forecast')
plt.legend()
plt.show()