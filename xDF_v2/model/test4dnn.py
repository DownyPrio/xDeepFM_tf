import tensorflow as tf
import numpy as np
from model import *

arr_1=np.array([[1,2,3,4,5]],dtype="float64")
INPUT=tf.constant(arr_1)

DNN_LIST=[Layer.DNN_Layer(5,"relu"),Layer.DNN_Layer(5,"relu"),Layer.DNN_Layer(1,"sigmoid")]

res=DNN.DNN_model(DNN_LIST).predict(INPUT)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(res))