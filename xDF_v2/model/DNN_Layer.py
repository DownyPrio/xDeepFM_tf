import tensorflow as tf
import numpy as np

class DNN_Layer:
    def __init__(self,units,activation):
        self.units=units
        self.activation=activation