import tensorflow as tf
import numpy as np

class LR_model():
    def __init__(self):
        return
    def predict(self,input):
        lr_model=tf.layers.Dense(units=1)
        #print(type(lr_model))
        lr_result=lr_model(input)
        return lr_result