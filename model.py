import tensorflow as tf
import pandas as pd


class MLPNet(tf.keras.Model):
    def __init__(self,vocab_size,embedding_dimension,block_n):
        super(MLPNet,self).__init__()
        self.embed_layer = tf.keras.layers.Embedding(vocab_size,embedding_dimension)
        self.stacked_MLPBlock = tf.keras.Sequential([MLPBlock() for n in range(block_n)])#tf.keras.Sequential([MLPBlock() for n in range(3)])

    def call(self,x):
        x = self.embed_layer(x) # (bs,512,768)
        x = self.stacked_MLPBlock(x)

        return x

class MLPBlock(tf.keras.Model):
    def __init__(self):
        super(MLPBlock,self).__init__()
        self.conv = tf.keras.Sequential([tf.keras.layers.Conv2D(5,kernel_size=2),
                                         tf.keras.layers.Conv2D(1,kernel_size=5)])

        self.dense_wise = tf.keras.Sequential([
                                               tf.keras.layers.Dense(768*4,activation='relu'),
                                               tf.keras.layers.Dense(768/2,activation='relu'),
                                               tf.keras.layers.Dense(768,activation='relu'),
                                               tf.keras.layers.LayerNormalization()])
        self.feature_wise = tf.keras.Sequential([
                                                 tf.keras.layers.Dense(512*4,activation='relu'),
                                                 tf.keras.layers.Dense(512/2,activation='relu'),       
                                                 tf.keras.layers.Dense(512,activation='relu'),
                                                 tf.keras.layers.LayerNormalization()])

    
    def call(self,x): #(bs,512,768)

        x1 , x2 ,x3 = tf.split(x,3,2)
        x = tf.stack([x1,x2,x3],-1) # cnn 용으로 (bs,512,768) -> (bs,3,512,-1)
        x = self.conv(x)
        x = tf.squeeze(x)

        x = self.dense_wise(x)
        x = tf.transpose(x,[0,2,1])

        x = self.feature_wise(x)
        x = tf.transpose(x,[0,2,1]) # 원상 복귀 

        return x
