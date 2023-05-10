#An experimental Keras SimpleImputer layer as a wrapper for sklearn's SimpleImputer

import tensorflow as tf
import pandas as pd
import numpy as np

class SimpleImputer(tf.keras.layers.Layer):
    def __init__(self,strategy='mean',**kwargs):
        from sklearn.impute import SimpleImputer
        super().__init__(**kwargs)
        self.strategy = strategy
        self.imputer = SimpleImputer(strategy=strategy)
        self.result = None
    
    def call(self,data):
        self.result = self.imputer.fit(data)
        self.result = self.imputer.transform(data)
        self.result = tf.convert_to_tensor(self.result)
        return self.result
        '''
        data = tf.convert_to_tensor(data)
        allowed_strategies = ["mean", "median", "most_frequent", "constant"]
        if(self.strategy=='constant'):
            pass
        elif(self.strategy=='mean'):
            pass
        elif(self.strategy=='median'):
            pass
        elif(self.strategy=='most_frequent'):
            pass
        else:
            raise Exception('Can only use these strategies:%s' %allowed_strategies)
        '''
    def get_config(self):
        config = super(SimpleImputer,self).get_config()
        config.update({
            "strategy": self.strategy
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
print(x.call(df))