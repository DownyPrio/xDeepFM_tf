import tensorflow as tf
import numpy as np
from model import *
import random
import time

list_p=[1]*500
list_1=list(map(lambda x:x*random.randrange(1,100),list_p))

list_2=list(map(lambda x:x*random.randrange(1,100),list_p))


trainSet=np.array([list_1,list_2]).T.reshape((500,1,2))

print(trainSet.shape)
#
# print(trainFinalSet)
w=np.array([[3],[4]])
b=np.array([[100]])
n=np.array([[1,2]]).astype('float64')
labelSet=((np.matmul(trainSet,w)+b)/800).astype("float64").reshape((-1,1,1))
print(labelSet.shape)


input=np.array([[[1,2,3]],
                [[2,4,6]],
                [[3,6,9]]]).astype("float64")
label=np.array([[[0.1]],
                [[0.2]],
                [[0.3]]]).astype("float64")

lr=LR.LR_model()
dnn=DNN.DNN_model([DNN_Layer.DNN_Layer(5,"relu"),DNN_Layer.DNN_Layer(5,"relu")])#,DNN_Layer.DNN_Layer(1,"sigmoid")])
cin=CIN.CIN_trans([CIN_Layer.CIN_layer(2),CIN_Layer.CIN_layer(2)])
List=[lr,dnn,cin]
start = time.clock()


model=xdf.XDF_model(List)
model.compile(trainSet,labelSet)
# model.predict(trainSet)
model.fit(trainSet,labelSet,100)
#当中是你的程序

elapsed = (time.clock() - start)
print("Time used:",elapsed)