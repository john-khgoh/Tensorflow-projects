import os
import pandas as pd
import numpy as np
from keras.layers import Dense, Flatten, LSTM, Activation
from keras.layers import Dropout, RepeatVector, TimeDistributed
from keras import Input, Model
import tensorflow as tf
#from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin

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

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
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

wd = os.getcwd()
checkpoint_filepath = wd + '//tmp//checkpoint//model.ckpt'
train_dir = wd + '//jsb_chorales//train//'
valid_dir = wd + '//jsb_chorales//valid//'
test_dir = wd + '//jsb_chorales//test//'

train_files = os.listdir(train_dir)
#train_files = train_files[:10]

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_mae',
    mode='min',
    save_best_only=True)

mae_list = []
split_time_ratio = 0.50 #The ratio between training and validation data
shuffle_buffer = 128 #Doesn't seem very important. Just ensure it's a large number. 
batch_size = 64 #Just the ordinary neural network batch size i.e. how many training examples to train at once 
window_size = 4 #Window size should approximate the periodicity of the waveform e.g. if it repeats every 7 days, the window should be 7
epochs = 20
learning_rate = 0.0007
input_dims = 4
output_dims = 4 # number of classes
lstm_layer = 12

for run_cnt,file in enumerate(train_files):
    df = pd.read_csv(train_dir + train_files[run_cnt])
    time = np.arange(len(df),dtype="int32")
    split_time = int(split_time_ratio * len(time)) #The cutoff point between training and validation data
    series = df
    time_train = time[:split_time]
    time_valid = time[split_time:]
    series_train = series[:split_time]
    series_valid = series[split_time:]
    series_valid_bkp = series_valid
    seq_length = len(series_train)

    #Normalization
    series_mean = np.mean(series)
    train_scaler = StandardScaler().fit(series_train)
    series_train = train_scaler.transform(series_train)
    valid_scaler = StandardScaler().fit(series_valid)
    series_valid = valid_scaler.transform(series_valid)

    #print(series_train)

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
    
    #Build model on first iteration
    if(run_cnt==0):
        model1 = build_model()
    #Load model on subsequent iterations from ModelCheckpoint
    else:
        model1.load_weights(checkpoint_filepath)
        
    history = model1.fit(train_ds,epochs=epochs,callbacks=[model_checkpoint_callback],validation_data=valid_ds)
       
    forecast_series = series[split_time - window_size:-1]

    rnn_forecast = model_forecast(model1, forecast_series, window_size)
    rnn_forecast = valid_scaler.inverse_transform(rnn_forecast)
    
    print(np.shape(rnn_forecast))
    print(np.shape(series_valid_bkp))
    mae_rnn = np.mean(tf.keras.metrics.mean_absolute_error(series_valid_bkp, rnn_forecast).numpy())
    mae_list.append(round(mae_rnn))
    print(mae_list)

print(mae_list)