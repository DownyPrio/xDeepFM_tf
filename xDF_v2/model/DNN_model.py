import tensorflow as tf
import numpy as np

class DNN_model:
    def __init__(self,layerList):
        self.layerList=[]
        for each in layerList:
            if each.activation=="relu":
                activation=tf.nn.relu
            elif each.activation=="tanh":
                activation=tf.nn.tanh
            elif each.activation=="sigmoid":
                activation=tf.nn.sigmoid
            tmp_layer=tf.layers.Dense(units=each.units,activation=activation)
            self.layerList.append(tmp_layer)
    def predict(self,input):
        tmp_res=input
        for each in self.layerList:
            tmp_res=each(tmp_res)
        return tmp_res