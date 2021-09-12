# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 22:08:16 2021

@author: Xiaohuaile
"""

import tensorflow.keras as keras
import tensorflow as tf
'''
stateful modules for real time processing
'''
class DprnnBlock_stateful(keras.layers.Layer):
    
    def __init__(self, numUnits, batch_size, L, width, channel, causal = True, **kwargs):
        super(DprnnBlock_stateful, self).__init__(**kwargs)

        self.numUnits = numUnits
        self.batch_size = batch_size
        self.causal = causal
        self.intra_rnn = keras.layers.Bidirectional(keras.layers.LSTM(units=self.numUnits//2, return_sequences=True,implementation = 2,recurrent_activation = 'hard_sigmoid',unroll = True))

        self.intra_fc = keras.layers.Dense(units = self.numUnits,)

        if self.causal:
            self.intra_ln = keras.layers.LayerNormalization(center=True, scale=True, axis = [-1,-2])
        else:
            self.intra_ln = keras.layers.LayerNormalization(center=False, scale=False)

        self.inter_rnn = keras.layers.LSTM(units=self.numUnits, return_sequences=True,implementation = 2,recurrent_activation = 'hard_sigmoid',unroll = True, stateful = False, return_state=True)
        
        self.inter_fc = keras.layers.Dense(units = self.numUnits,) 

        if self.causal:
            self.inter_ln = keras.layers.LayerNormalization(center=True, scale=True, axis = [-1,-2])
        else:
            self.inter_ln = keras.layers.LayerNormalization(center=False, scale=False)

        self.L = L
        self.width = width
        self.channel = channel
        
        
    def call(self, x):
        # Intra-Chunk Processing
        batch_size = self.batch_size
        L = self.L
        width = self.width
        
        intra_rnn = self.intra_rnn
        intra_fc = self.intra_fc
        intra_ln = self.intra_ln
        inter_rnn = self.inter_rnn
        inter_fc = self.inter_fc
        inter_ln = self.inter_ln
        channel = self.channel
        causal = self.causal
        h_state = tf.Variable(tf.zeros([batch_size,width,channel]),trainable = False,)
        c_state = tf.Variable(tf.zeros([batch_size,width,channel]),trainable = False,)
        
        # input shape (bs,T,F,C) --> (bs*T,F,C)
        intra_LSTM_input = tf.reshape(x,[-1,width,channel])
        # (bs*T,F,C)
        intra_LSTM_out = intra_rnn(intra_LSTM_input)
        
        # (bs*T,F,C) channel axis dense
        intra_dense_out = intra_fc(intra_LSTM_out)
            
        if causal:
            # (bs*T,F,C) --> (bs,T,F,C) Freq and channel norm
            intra_ln_input = tf.reshape(intra_dense_out,[batch_size,-1,width,channel])
            intra_out = intra_ln(intra_ln_input)
            
        else:       
            # (bs*T,F,C) --> (bs,T*F*C) global norm
            intra_ln_input = tf.reshape(intra_dense_out,[batch_size,-1])
            intra_ln_out = intra_ln(intra_ln_input)
            intra_out = tf.reshape(intra_ln_out,[batch_size,L,width,channel])
            
        # (bs,T,F,C)
        intra_out = keras.layers.Add()([x,intra_out])
        #%%
        # (bs,T,F,C) --> (bs,F,T,C)
        inter_LSTM_input = tf.transpose(intra_out,[0,2,1,3])
        # (bs,F,T,C) --> (bs*F,T,C)
        inter_LSTM_input = tf.reshape(inter_LSTM_input,[batch_size*width,L,channel])

        inter_LSTM_out, inter_LSTM_h, inter_LSTM_c = inter_rnn(inter_LSTM_input,initial_state = [h_state[0],c_state[0]])
        
        # (bs,F,T,C) Channel axis dense
        inter_dense_out = inter_fc(inter_LSTM_out)
        
        inter_dense_out = tf.reshape(inter_dense_out,[batch_size,width,L,channel])
        
        if causal:
            # (bs,F,T,C) --> (bs,T,F,C)
            inter_ln_input = tf.transpose(inter_dense_out,[0,2,1,3])
            inter_out = inter_ln(inter_ln_input)
            
        else:
            # (bs,F,T,C) --> (bs,F*T*C)
            inter_ln_input = tf.reshape(inter_dense_out,[batch_size,-1])
            inter_ln_out = inter_ln(inter_ln_input)
            inter_out = tf.reshape(inter_ln_out,[batch_size,width,L,channel])
            # (bs,F,T,C) --> (bs,T,F,C)
            inter_out = tf.transpose(inter_out,[0,2,1,3])
        # (bs,T,F,C)
        inter_out = keras.layers.Add()([intra_out,inter_out])

        with tf.control_dependencies([inter_LSTM_h,inter_LSTM_c]):
            self.update_h = h_state.assign(tf.expand_dims(inter_LSTM_h,axis = 0),read_value = False)
            self.update_c = c_state.assign(tf.expand_dims(inter_LSTM_c,axis = 0),read_value = False)
        
        self.reset_h = h_state.assign(tf.zeros([batch_size,width,channel]),read_value = False)
        self.reset_c = c_state.assign(tf.zeros([batch_size,width,channel]),read_value = False)
                                 
        tf.add_to_collection('upop', [self.update_h,self.update_c])
        tf.add_to_collection('rsop', [self.reset_h,self.reset_c])
        
        return inter_out

'''
stateful Conv2D inherit from base classes Conv2D
'''
class Conv2D_stateful(keras.layers.Conv2D):
    def __init__(self, filters, keranel_size, strides, padding,state_shape, *args, **kwargs):
        self.state_shape = state_shape
        super().__init__(filters, keranel_size, strides, padding,*args, **kwargs)
    
    # Override call
    def call(self, x):

        padding_state = tf.Variable(tf.zeros(self.state_shape), trainable=False)
        padding_data = tf.concat([padding_state, x], axis=1)
        output = super().call(padding_data)
        
        with tf.control_dependencies([padding_data,output]):
            self.update_op = padding_state.assign(x, read_value=False)
            
        self.reset = padding_state.assign(tf.zeros(self.state_shape), read_value=False)
        tf.add_to_collection('upop', self.update_op)
        tf.add_to_collection('rsop', self.reset)
        return output


'''
stateful Conv2DTranspose inherit from base classes Conv2DTranspose
'''
class DeConv2D_stateful(keras.layers.Conv2DTranspose):
    
    def __init__(self, filters, keranel_size, strides, padding, state_shape, *args,**kwargs):

        self.state_shape = state_shape
        self.padding = padding        
        super().__init__(filters, keranel_size, strides, padding,*args, **kwargs)
        
    # Override call
    def call(self, x):
        
        padding_state = tf.Variable(tf.zeros(self.state_shape), trainable=False) 
        padding_data = tf.concat([padding_state,x],axis = 1)
        output = super().call(padding_data)
        
        if self.padding == 'same':
            output = output[:,-1:,:,:]
            
        with tf.control_dependencies([padding_data,output]):
            self.update_op = padding_state.assign(x, read_value=False)
        
        self.reset = padding_state.assign(tf.zeros(self.state_shape), read_value=False)
        
        tf.add_to_collection('upop', self.update_op)
        tf.add_to_collection('rsop', self.reset)
        return output
