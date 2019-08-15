import tensorflow as tf
import numpy as np


class DNN_model:
    def __init__(self):
        self.layerList=[]

    def append(self,DNN_Layer):
        self.layerList.append(DNN_Layer.layer)
    def predict(self,input):
        tmp_res=input
        for each in self.layerList:
            print(each)
            tmp_res=each(tmp_res)
        dnn_res=tmp_res
        return dnn_res


# def dnn_model(input):
#     layer_1=tf.layers.Dense(10, activation=tf.nn.relu)
#     layer_2=tf.layers.Dense(10,activation=tf.nn.relu)
#     #output_layer=tf.layers.Dense(1,activation=tf.nn.sigmoid)
#     layer_1_res=layer_1(input)
#     layer_2_res=layer_2(layer_1_res)
#     output_res=layer_2_res
#     #output_res=output_layer(layer_2_res)
#
#     return output_res
class DNN_Layer:
    def __init__(self,nodes,activation):
        if activation=="relu":
            self.activation=tf.nn.relu
        elif activation=="tanh":
            self.activation=tf.nn.tanh
        elif activation=="sigmoid":
            self.activation=tf.nn.sigmoid
        self.layer=tf.layers.Dense(nodes, activation=self.activation)