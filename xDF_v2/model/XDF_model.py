from model import *
import tensorflow as tf
import numpy as np

class XDF_model:
    def __init__(self,modelList):
        self.lr=modelList[0]
        self.dnn=modelList[1]
        self.cin=modelList[2]
    #input:array
    def compile(self,input,label):
        d0=input.shape[0]
        d1=input.shape[1]
        d2=input.shape[2]
        #print("label:{},{}".format(label.shape[1],label.shape[2]))
        self.x=tf.placeholder(dtype=tf.float64,shape=(None,d1,d2))
        self.y=tf.placeholder(dtype=tf.float64,shape=(None,label.shape[1],label.shape[2]))
        lr_res=self.lr.predict(self.x)
        dnn_res=self.dnn.predict(self.x)
        _,cin_res=CIN.CIN_model(self.cin.LayerList,self.x,input).predict(self.x)
        res=lr_res
        res=tf.concat([res,dnn_res],axis=2)
        res=tf.concat([res,cin_res],axis=2)
        out_layer=tf.layers.Dense(units=1,activation=tf.nn.sigmoid)
        out_res=out_layer(res)
        self.compile_res=out_res
    def predict(self,input):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print("final res:")
            print(sess.run(self.compile_res,feed_dict={self.x:input}))
    def fit(self,train,label,epochs):
        y_pred=self.compile_res
        # print(y_pred)
        # print(self.y)
        #losses=tf.losses.mean_squared_error(predictions=y_pred,labels=self.y)
        losses=tf.losses.log_loss(predictions=y_pred,labels=self.y)
        opt=tf.train.AdamOptimizer(0.000001)#GradientDescentOptimizer(0.000001)
        train_fit=opt.minimize(losses)
        with tf.Session() as sess:
            for i in range(epochs):
                sess.run(tf.global_variables_initializer())
                print("epochs:{}/{}".format(i,epochs))
                sess.run(train_fit,feed_dict={self.x:train,self.y:label})
                print("losses is:")
                print(sess.run(losses,feed_dict={self.x:train,self.y:label}))
            
        