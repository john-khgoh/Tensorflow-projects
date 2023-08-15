from os import getcwd, listdir
import pandas as pd
import tensorflow as tf

#Loading the data
wd = getcwd()
file = wd + '\\energy_dataset.csv'
df = pd.read_csv(file)

#Splitting the data into train and validation
def train_validation_split(time,series,split):
    train_time = time[:split]
    valid_time = time[split:]
    train_series = series[:split]
    valid_series = series[split:]
    return train_time,valid_time,train_series,valid_series

len_df = len(df)
split = int(0.8 * len_df)
time = df['time']
series = df['price actual']

train_time, valid_time, train_series, valid_series = train_validation_split(time,series,split)

window_size_const = 64
batch_size_const = 32
shuffle_buffer_const = 1000

def windowed_dataset(series,window_size,batch_size,shuffle_buffer):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1,shift=1,drop_remainder=True)
    ds = ds.flat_map(lambda w:w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1],w[-1]))
    ds = ds.batch(batch_size).prefetch(1)
    return ds

train_set = windowed_dataset(train_series,window_size=window_size_const,batch_size=batch_size_const,shuffle_buffer=shuffle_buffer_const)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=32,kernel_size=5,strides=1,padding='causal',activation='relu',input_shape=[None,1]),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32,return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(30,activation='relu'),
    tf.keras.layers.Dense(10,activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(loss=tf.keras.losses.Huber(),optimizer=tf.keras.optimizers.Adam(),metrics=['mae'])
history = model.fit(train_set,epochs=20)