import tensorflow as tf
import numpy as np

class LR_model:
    def __init__(self):
        pass

    def predict(self,input):
        linear_model=tf.layers.Dense(units=1)
        lr_res=linear_model(input)
        return lr_res