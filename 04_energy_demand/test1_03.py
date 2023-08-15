from os import getcwd, listdir
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer

#from google.colab import drive
#drive.mount('/content/gdrive')

#Loading the data
#file = '/kaggle/input/energy-demand/energy_dataset.csv'
#output = '/kaggle/working/'
wd = getcwd()
file = wd + '\\energy_dataset.csv'
df = pd.read_csv(file)

#Splitting the data into train and validation
def train_validation_split(time,series,split):
    time_train = time[:split]
    time_valid = time[split:]
    series_train = series[:split]
    series_valid = series[split:]
    return time_train,time_valid,series_train,series_valid

len_df = len(df)
split = int(0.8 * len_df)
time = df['time']

imputer = SimpleImputer(strategy='median')
imputer.fit(df[['price actual','total load actual']])
series = imputer.transform(df[['price actual','total load actual']])
#series = df[['price actual','total load actual']]
series = pd.DataFrame(series,columns=['price actual','total load actual'])
series_mean_list = list(series.mean())
no_of_cols = series.shape[1]


time_train, time_valid, series_train, series_valid = train_validation_split(time,series,split)

window_size = 32
batch_size = 64
shuffle_buffer = 30000
epochs = 20
learning_rate = 5e-4

train_ds = tf.keras.utils.timeseries_dataset_from_array(
    series_train,
    targets = series_train['price actual'],
    sequence_length = window_size,
    batch_size = batch_size,
    shuffle = True
)
valid_ds = tf.keras.utils.timeseries_dataset_from_array(
    series_valid,
    targets = series_valid['price actual'],
    sequence_length = window_size,
    batch_size = batch_size,
    shuffle = True
)

'''
def windowed_dataset(series,window_size,batch_size,shuffle_buffer):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1,shift=1,drop_remainder=True)
    ds = ds.flat_map(lambda w:w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1],w[-1]))
    ds = ds.batch(batch_size).prefetch(1)
    return ds

train_set = windowed_dataset(series_train,window_size=window_size,batch_size=batch_size,shuffle_buffer=shuffle_buffer)
'''
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=32,kernel_size=5,strides=1,padding='causal',activation='relu',input_shape=[None,no_of_cols]),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(12,return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(12)),
    tf.keras.layers.Dense(24,activation='relu'),
    tf.keras.layers.Dense(12,activation='relu'),
    tf.keras.layers.Dense(1,activation='relu')
])

model.compile(loss=tf.keras.losses.Huber(),optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),metrics=['mae'])
history = model.fit(train_ds,epochs=epochs)

def model_forecast(model,series,window_size):
  ds = tf.data.Dataset.from_tensor_slices(series)
  ds = ds.window(window_size,shift=1,drop_remainder=True)
  ds = ds.flat_map(lambda w:w.batch(window_size))
  ds = ds.batch(batch_size).prefetch(1)
  forecast = model.predict(ds)
  return forecast

forecast_series = series[split - window_size:-1]
rnn_forecast = model_forecast(model,forecast_series,window_size).squeeze()

naive_forecast = np.array([series_mean_list[0]] * len(series_valid))
mae_naive = tf.keras.metrics.mean_absolute_error(series_valid['price actual'],naive_forecast).numpy()
mae_rnn = tf.keras.metrics.mean_absolute_error(series_valid['price actual'],rnn_forecast).numpy()
print(mae_naive,mae_rnn)

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

plt.figure(figsize=(10, 6))
plot_series(time_valid, series_valid)
plt.savefig(output + '1.png')

plot_series(time_valid, rnn_forecast)
plt.savefig(output + '2.png')