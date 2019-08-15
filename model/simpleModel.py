import tensorflow as tf
import numpy as np

# x=tf.placeholder()
# y=tf.placeholder()
n=np.array([[1,2]]).astype('float32')
a=tf.constant(n)

#lr part
def lr_model(input):
    lr_model=tf.layers.Dense(units=1)
    print(type(lr_model))
    lr_result=lr_model(input)
    with tf.Session() as sess:
        init=tf.initialize_all_variables()
        sess.run(init)
        print(sess.run(lr_result))
    return lr_result


lr_model(n)
#dnn_part
def dnn_model(input):
    layer_1=tf.layers.Dense(10, activation=tf.nn.relu)
    layer_2=tf.layers.Dense(10,activation=tf.nn.relu)
    #output_layer=tf.layers.Dense(1,activation=tf.nn.sigmoid)
    layer_1_res=layer_1(input)
    layer_2_res=layer_2(layer_1_res)
    output_res=layer_2_res
    #output_res=output_layer(layer_2_res)
    with tf.Session() as sess_2:
        init=tf.initialize_all_variables()
        sess_2.run(init)
        print("************")
        print(sess_2.run(layer_1_res))
        print("************")
        print(sess_2.run(layer_2_res))
        print("*************")
        print(sess_2.run(output_res))

    return output_res


print("************")
dnn_model(n)
print("*************")

# def Dicar(W1,W2):
#
#     for index in range(W1.get_shape()[0]):
#         print(index)
#
#         app_val=tf.constant(W1[index]*W2[index])
#         if index==0:
#             res_val=tf.make_ndarray([app_val])
#         else:
#             res_val=tf.concat(res_val,[app_val],axis=0)
#     return res_val
def Interaction(Xn,X0):
    #Xn:(h,D)
    #X0:(m,D)

    for index in range(Xn.get_shape()[1]):

        # f=tf.transpose(tf.transpose(Xn)[index])
        # p=tf.print(f,[f.shape])
        # print(tf.print(tf.transpose(X0)[index]))
        X_tmp=tf.matmul(tf.transpose([Xn[:,index]]),[X0[:,index]])
        X_tmp=tf.reshape(X_tmp,(1,-1))
        if index==0:
            X_tmp_list=X_tmp
        else:
            # with tf.Session() as sess_5:
            #     print(sess_5.run(X_tmp_list))
            #     print(sess_5.run(X_tmp))
            X_tmp_list=tf.concat([X_tmp_list,X_tmp],axis=0)
    #X_tmp_list.shape=((D,m*m))
    return X_tmp_list

class CIN_Layer:
    def __init__(self,H,H_per,X0):
        self.H=H
        self.X0=X0
        self.H0=X0.get_shape()[0]
        self.Weights=tf.Variable(np.ones((H_per*self.H0,H)))
    def forward(self,tensor):
        Inter_res=Interaction(tensor,self.X0)
        with tf.Session() as sess_6:
            sess_6.run(tf.initialize_all_variables())
            print(sess_6.run(Inter_res))
            print(sess_6.run(self.Weights))
        Outer_res=tf.matmul(Inter_res,self.Weights)#((D,H))
        Outer_vector=tf.reduce_sum(Outer_res,axis=0)
        return Outer_res,Outer_vector

def CIN_model(input):
    a=input
    l_1=CIN_Layer(2,1,a)
    l_1_res,l_1_vec=l_1.forward(a)
    l_1_res=tf.transpose(l_1_res)
    l_2=CIN_Layer(3,2,a)
    l_2_res,l_2_vec=l_2.forward(l_1_res)
    # with tf.Session() as sess_7:
    #     sess_7.run(tf.initialize_all_variables())
    #     print(sess_7.run((l_1_vec,l_2_vec)))
    output_vector=tf.concat([l_1_vec,l_2_vec],axis=0)
    return output_vector
    # init=tf.initialize_all_variables()
    # with tf.Session() as sess_3:
    #     sess_3.run(init)
    #     print(sess_3.run(output_vector))

# input_array=np.array([[1,0,0,0,1,0,0,1]]).astype("float64")
# input=tf.constant(input_array)
#
# lr_res=lr_model(input)
# dnn_res=dnn_model(input)
# cin_res=CIN_model(input)
#
#
# lr_dnn_res=tf.concat([lr_res,dnn_res],axis=1)
# res=tf.concat([lr_dnn_res,[cin_res]],axis=1)
#
# final_lr=tf.layers.Dense(units=1,activation=tf.nn.sigmoid)
# result_final=final_lr(res)
# with tf.Session() as sess_8:
#     sess_8.run(tf.initialize_all_variables())
#     print(sess_8.run(result_final))






