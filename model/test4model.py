from model import *
import tensorflow as tf
import numpy as np

import random
list_p=[1]*500
list_1=list(map(lambda x:x*random.randrange(1,100),list_p))

list_2=list(map(lambda x:x*random.randrange(1,100),list_p))


trainSet=np.array([list_1,list_2]).T
trainFinalSet=[]
for index in range(len(trainSet)):
    trainFinalSet.append(np.array([trainSet[index]]))
print(trainSet.shape)
#
# print(trainFinalSet)
w=np.array([[3],[4]])
b=np.array([[100]])
n=np.array([[1,2]]).astype('float64')
labelSet=((np.matmul(trainFinalSet,w)+b)/800).astype("float64").reshape((-1,1))
print(labelSet.max())
trainFinalSet=np.array(trainFinalSet).astype("float64").reshape(-1,2)
x=tf.placeholder(tf.float64,(None,2))
y=tf.placeholder(tf.float64,(None,1))

lr_model=LR.LR_model()
dnn_model=DNN.DNN_model()
l_1=DNN.DNN_Layer(10,"relu")
l_2=DNN.DNN_Layer(10,"relu")
dnn_model.append(l_1)
dnn_model.append(l_2)
# la=dnn_model.predict(trainFinalSet)
# with tf.Session() as se:
#     se.run(tf.initialize_all_variables())
#     print(se.run(la))

#
q=lr_model.predict(n)
e=dnn_model.predict(n)
with tf.Session() as se:
    se.run(tf.initialize_all_variables())
    print(se.run(q))
    print(se.run(e))
input=np.array([[1,2]]).astype("float64")
c_1=CIN.CIN_Layer(3)
c_2=CIN.CIN_Layer(2)
list_cin=[]
list_cin.append(c_1)
list_cin.append(c_2)
cin_model=CIN.CIN_model([Layer.CIN_layer(3,input_shape=(2,1,3)),Layer.CIN_layer(2)],x,trainSet)
# la1=cin_model.predict()
# with tf.Session() as se:
#     print("*(((((((((((((((((")
#     se.run(tf.initialize_all_variables())
#     print(se.run(la1))
model_list=[lr_model,dnn_model,cin_model]

model=xdfm.xDeepFM_model(model_list)
y_pred=model.predict(x)
# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#     print(sess.run(y_pred))
losses=tf.losses.log_loss(y_pred,y)
optimizer=tf.train.GradientDescentOptimizer(0.0001)
train=optimizer.minimize(losses)
init=tf.initialize_all_variables()
print(trainFinalSet.shape)
print(labelSet.shape)
with tf.Session() as sess:
    sess.run(init)
    epochs=1000
    print(sess.run(tf.shape(y_pred)))
    for i in range(epochs):
        print("epoch:  {}/{}".format(i+1,epochs))
        _=sess.run(train,feed_dict={x:trainFinalSet,y:labelSet})
        losses=sess.run(losses,feed_dict={x:trainFinalSet,y:labelSet})
        print("loss:{}".format(losses))
