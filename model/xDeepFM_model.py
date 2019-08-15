import tensorflow as tf
import numpy as np

class xDeepFM_model:
    def __init__(self,model_list):
        self.model_list=model_list
    def predict(self,input):
        lr_res=self.model_list[0].predict(input)
        dnn_res=self.model_list[1].predict(input)
        cin_res=self.model_list[2](input)
        # with tf.Session() as sess:
        #     sess.run(tf.initialize_all_variables())
        #     print("lr:")
        #     print(sess.run(lr_res))
        #     print("dnn")
        #     print(sess.run(dnn_res))
        lr_dnn_res=tf.concat([lr_res,dnn_res],axis=1)
        res=tf.concat([lr_dnn_res,[cin_res]],axis=1)

        final_lr=tf.layers.Dense(units=1,activation=tf.nn.sigmoid)
        result_final=final_lr(res)
        return result_final