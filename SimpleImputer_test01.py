#An experimental Keras SimpleImputer layer

import tensorflow as tf
import pandas as pd
import numpy as np

class SimpleImputer(tf.keras.layers.Layer):
    def __init__(self,strategy='mean',**kwargs):
        super().__init__(**kwargs)
        self.strategy = strategy
    
    def call(self,data):
        data = tf.convert_to_tensor(data)
        print(data)
        if(self.strategy=='constant'):
            pass
        elif(self.strategy=='mean'):
            pass
        elif(self.strategy=='median'):
            pass
        elif(self.strategy=='most_frequent'):
            pass
        else:
            raise Error()
    
    def get_config(self):
        config = super(SimpleImputer,self).get_config()
        config.update({
        })
        return config

#Testing with dataframes        
a = [1,2,3,4,5,np.nan]
b = [3,4,5,6,np.nan,7]
a_df = pd.DataFrame(a,columns=['a'])
b_df = pd.DataFrame(b,columns=['b'])
df = pd.concat([a_df,b_df],axis=1)

#Testing with numpy arrays
c = np.random.randn(2,5)

x = SimpleImputer('mean')
x.call(c)