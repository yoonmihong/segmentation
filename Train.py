#!/usr/bin/env python



# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 10:12:52 2016

@author: tmquan
"""
from Model 		import *

from TFlearn	import *
from Utility	import *


######################################################################################

def train():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    from tensorflow.python.client import device_lib
    import tensorflow as tf

    from tensorflow.python.ops import control_flow_ops
    tf.python.control_flow_ops = control_flow_ops

    X = np.load('./data/3_npy/train_image.npy')
    y = np.load('./data/3_npy/train_label.npy')


    print('Shuffle data...')
    X, y = shuffle(X, y)

    X = X.astype(np.float32)
    y = y.astype(np.float32)

    print X.shape
    print y.shape


    # Get the model from Model.py
    model = get_model()
    # Shuffle the data
        
    model.fit(X, y, run_id="WHS_20171026",
              n_epoch=150,
              validation_set=0.1,
              shuffle=True,
              show_metric=True,
              snapshot_epoch=True,
              batch_size=15)
    model.save('MUNet_depth5_model_WHS')
    
    #model.fit(X, y, run_id="fully_convolutional_neural_network",
    #          n_epoch=100,
    #          validation_set=0.2,
    #          shuffle=True,
    #          show_metric=True,
    #          snapshot_step=100,
    #          snapshot_epoch=True,
    #          batch_size=20)

if __name__ == '__main__':
	train()
