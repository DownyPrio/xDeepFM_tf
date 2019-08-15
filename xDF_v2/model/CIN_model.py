import tensorflow as tf
import numpy as np
class CIN_trans:
    def __init__(self,LayerList):
        self.LayerList=LayerList

class CIN_model:
    def __init__(self,LayerList,input,arr):
        self.layerList=LayerList
        self.input=input
        self.samples=arr.shape[0]
        self.h0=arr.shape[1]
        self.d=arr.shape[2]
        #self.x=tf.placeholder(shape=(None,self.h0,self.d),dtype=tf.float64)
        self.m0=self.h0
        #self.m0=self.layerList[0].input_shape[1]
        #self.d=self.layerList[0].input_shape[2]
        self.listParse()
        self.build_w()
        #self.describe()
    
    def __call__(self, *args, **kwargs):
        return
    def describe(self):
        print("H LIST:")
        print(self.H_list)
        print("W_row LIST:")
        print(self.W_row_list)
        print("W list:")
        for each in range(len(self.w)):
            print("layer {} shape is:".format(each))
            print(self.W_row_list[each],self.H_list[each])
    def build_w(self):
        self.w=[]
        for index in range(len(self.W_row_list)):
            # print(self.W_row_list[index])
            # print(self.H_list[index])
            tmp_w=tf.Variable(tf.ones((self.W_row_list[index],self.H_list[index]),dtype=tf.float64))
            self.w.append(tmp_w)
    def listParse(self):
        self.H_list=[]
        self.W_row_list=[]
        for index in range(len(self.layerList)):

            if index==0:
                self.W_row_list.append(self.m0*self.m0)
            else:
                self.W_row_list.append(self.H_list[-1]*self.m0)
            self.H_list.append(self.layerList[index].H)

    #######!!!!!!!!!!!!!!1
    def filer(self,Xn,X0):
        for index in range(self.d):
            # print("index:" )
            # print(index)
            mat_1=tf.convert_to_tensor(Xn[:,:,index])
            mat_2=X0[:,:,index]
            mat_2_T=tf.transpose(mat_2,perm=(0,2,1))
            mat_2_T=tf.reshape(mat_2,shape=())
            print(type(mat_1))
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                print(sess.run(mat_1))
                print(sess.run(mat_2_T))
            mat_d=tf.matmul(mat_1,mat_2_T)
            if index==0:
                mat=mat_d
            else:
                mat=tf.concat([mat,mat_d],axis=1)
        return mat

    def vec(self,mat):
        return tf.reduce_sum(mat,axis=0)
    def forward(self,tmp_mat,index):
        X2Bfilter=self.sliceAssemble(tmp_mat,self.input,self.d)
        # print("w shape:")
        # print(self.w[index])
        tmp_res=tf.matmul(X2Bfilter,self.w[index])
        #tmp_vec=self.vec(tmp_res)
        return tmp_res#,tmp_vec
    def predict(self,input):
        tmp_res=input
        vec_list=[]
        for index in range(len(self.layerList)):
            # print("layer {}".format(index))
            tmp_res=self.forward(tmp_res,index)
            tmp_vec=tf.reshape(tf.reduce_sum(tmp_res,axis=1),shape=(-1,1,self.H_list[index]))
            if index==0:
                out_vec=tmp_vec
            else:
                out_vec=tf.concat([out_vec,tmp_vec],axis=2)
            tmp_res=tf.transpose(tmp_res,perm=[0,2,1])
            #vec_list.append(tmp_vec)
        #out_res=vec_list[0]
        # for i in range(1,len(vec_list)):
        #     out_res=tf.concat([out_res,vec_list[i]],axis=1)
        return tmp_res,out_vec

    def sliceAssemble(self,Data1,Data2,d):
        for index in range(self.samples):
            # print(index)

            data1=tf.reshape(Data1[index],shape=(1,-1,d))
            data2=tf.reshape(Data2[index],shape=(1,-1,d))
            tmp_X=self.myMul(data1,data2,d)
            tmp_X=tf.reshape(tmp_X,shape=(1,d,-1))
            if index==0:
                XAssemble=tmp_X
            else:
                XAssemble=tf.concat([XAssemble,tmp_X],axis=0)
            # with tf.Session() as sess:
            #     sess.run(tf.global_variables_initializer())
            #     print("............")
            #     print(sess.run(XAssemble))
        return XAssemble


    #完成两个X间的特征交互
    #输入：两个shape为（h,d）的tensor
    #输出：一个shape为（d,h*h）的tensor
    def myMul(self,Xn,X0,d):
        X0_T=tf.transpose(X0,perm=(0,2,1))
        for index in range(d):
            bflatten=tf.matmul(tf.reshape(Xn[:,:,index],(-1,1)),X0_T[:,index,:])
            bflatten=tf.reshape(bflatten,(1,-1))
            if index==0:
                vec=bflatten
            else:
                vec=tf.concat([vec,bflatten],axis=0)
        return vec


