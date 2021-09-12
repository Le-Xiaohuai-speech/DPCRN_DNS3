# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 11:45:28 2021

@author: xiaohuaile
"""
from tensorflow.keras.callbacks import ReduceLROnPlateau, CSVLogger, \
    EarlyStopping, ModelCheckpoint
import tensorflow as tf

def reshape(x,axis):
    return tf.reshape(x,axis)

def transpose(x,axis):
    return tf.transpose(x,axis)

# checkpoints for parallel training
class ParallelModelCheckpoints(ModelCheckpoint): 
							
    def __init__(self, 
                    model, 
                    filepath='./log/epoch-{epoch:02d}_loss-{loss:.4f}_acc-{val_acc:.4f}_lr-{lr:.5f}.h5',
                    monitor='val_acc',
                    verbose=1,
                    save_best_only=True,
                    save_weights_only=False,
                    mode='auto',
                    period=1):
        self.single_model = model
        super(ParallelModelCheckpoints, self).__init__(filepath, monitor, verbose,save_best_only, save_weights_only, mode, period)

    def set_model(self, model):
        super(ParallelModelCheckpoints, self).set_model(self.single_model)