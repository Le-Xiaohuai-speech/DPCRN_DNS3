"""
Created on Tue Mar 24 10:48:07 2021
@author: Xiaohuaile
"""
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda, Input, BatchNormalization, Concatenate, LayerNormalization,PReLU

from tensorflow.keras import backend as K
import soundfile as sf
import librosa
from random import seed
import numpy as np

from tensorflow.keras import backend as K
'''
stateful modules used in the real time processing
'''
from stateful_modules import DprnnBlock_stateful, Conv2D_stateful, DeConv2D_stateful
import matplotlib.pyplot as plt
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
np.random.seed(4)
blockLen = 400
blockshift = 200        
win = np.sin(np.arange(.5,blockLen-.5+1)/blockLen*np.pi)
win = tf.constant(win,dtype = 'float32')  

input_norm = True

#使用cpu计算
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#overlapadd 函数
def overlapadd(frame,hop = 200):
    
    N_frame, L_frame = frame.shape
    length = L_frame + (N_frame - 1) * hop
    output = np.zeros(length)
    for i in range(N_frame):
        output[hop * i : hop * i + L_frame] += frame[i]
 
    return output 
#%%
#stft keras layer
def stftLayer(x, mode ='mag_pha',stateful = False):

    # creating frames from the continuous waveform
    if not stateful:
        frames = tf.signal.frame(x, blockLen, blockshift)
    else:
        frames = tf.expand_dims(x,axis =1)
        
    frames = win*frames
    # calculating the fft over the time frames. rfft returns NFFT/2+1 bins.
    stft_dat = tf.signal.rfft(frames)
    # calculating magnitude and phase from the complex signal
    output_list = []
    if mode == 'mag_pha':
        mag = tf.math.abs(stft_dat)
        phase = tf.math.angle(stft_dat)
        output_list = [mag, phase]
    elif mode == 'real_imag':
        real = tf.math.real(stft_dat)
        imag = tf.math.imag(stft_dat)
        output_list = [real, imag]            
    # returning magnitude and phase as list
    return output_list 
#ifft layer
def ifftLayer( x,mode = 'mag_pha'):

    if mode == 'mag_pha':
    # calculating the complex representation
        s1_stft = (tf.cast(x[0], tf.complex64) * 
                    tf.exp( (1j * tf.cast(x[1], tf.complex64))))
    elif mode == 'real_imag':
        s1_stft = tf.cast(x[0], tf.complex64) + 1j * tf.cast(x[1], tf.complex64)
    # returning the time domain frames
    return tf.signal.irfft(s1_stft)
#overlap add layer
def overlapAddLayer(x):

    # calculating and returning the reconstructed waveform
    x = x - tf.expand_dims(tf.reduce_mean(x,axis = -1),2)
    return tf.signal.overlap_and_add(x, blockshift)      
        
# make complex product
def mk_mask(x):
    [noisy_real,noisy_imag,mask] = x
    #noisy_real = noisy_real[:,:,:,0]
    #noisy_imag = noisy_imag[:,:,:,0]
    
    mask_real = mask[:,:,:,0]
    mask_imag = mask[:,:,:,1]
    
    enh_real = noisy_real * mask_real - noisy_imag * mask_imag
    enh_imag = noisy_real * mask_imag + noisy_imag * mask_real
    
    return [enh_real,enh_imag]

#%%
#hyperparameter in the network
CNN_filter_list = [32,32,32,64,128]
DeCNN_filter_list = [64,32,32,32,2]
CNN_state_list = [[1,1,201,2],[1,1,100,32],[1,1,50,32],[1,1,50,32],[1,1,50,64]]
DeCNN_state_list = [[1,1,50,256],[1,1,50,128],[1,1,50,64],[1,1,50,64],[1,1,100,64]]

#make model for real time
def mk_Encoder_RT():
    
    Encoder_input = Input(batch_shape=(1, 400))
    #time_frames = Lambda(self.sep2frame,name ='sep2frame_1')(time_dat)
    
    # calculate STFT
    real,imag = Lambda(stftLayer,arguments = {'mode':'real_imag','stateful':True})(Encoder_input)

    input_complex_spec = tf.stack([real,imag],axis=-1)
    
    if input_norm:    
        input_complex_spec = LayerNormalization(axis = [-1,-2])(input_complex_spec)
        
    name = 'encoder'
    conv_1 = Conv2D_stateful(filters = CNN_filter_list[0], keranel_size = (2,5), strides = (1,2), padding = [[0,0],[0,0],[0,2],[0,0]], state_shape = CNN_state_list[0],name = name+'_conv_1')(input_complex_spec)
    bn_1 = BatchNormalization(name = name+'_bn_1')(conv_1)
    out_1 = PReLU(shared_axes=[1,2])(bn_1)
    
    conv_2= Conv2D_stateful(filters = CNN_filter_list[1], keranel_size = (2,3), strides = (1,2), padding = [[0,0],[0,0],[0,1],[0,0]], state_shape = CNN_state_list[1],name = name+'_conv_2')(out_1)
    bn_2 = BatchNormalization(name = name+'_bn_2')(conv_2)
    out_2 = PReLU(shared_axes=[1,2])(bn_2)
    
    conv_3 = Conv2D_stateful(filters = CNN_filter_list[2], keranel_size = (2,3), strides = (1,1), padding = [[0,0],[0,0],[1,1],[0,0]], state_shape = CNN_state_list[2],name = name+'_conv_3')(out_2)
    bn_3 = BatchNormalization(name = name+'_bn_3')(conv_3)
    out_3 = PReLU(shared_axes=[1,2])(bn_3)
    
    conv_4= Conv2D_stateful(filters = CNN_filter_list[3], keranel_size = (2,3), strides = (1,1), padding = [[0,0],[0,0],[1,1],[0,0]], state_shape = CNN_state_list[3],name = name+'_conv_4')(out_3)
    bn_4 = BatchNormalization(name = name+'_bn_4')(conv_4)
    out_4 = PReLU(shared_axes=[1,2])(bn_4)
    
    conv_5 = Conv2D_stateful(filters = CNN_filter_list[4], keranel_size = (2,3), strides = (1,1), padding = [[0,0],[0,0],[1,1],[0,0]], state_shape = CNN_state_list[4],name = name+'_conv_5')(out_4)
    bn_5 = BatchNormalization(name = name+'_bn_5')(conv_5)
    out_5 = PReLU(shared_axes=[1,2])(bn_5)
    
    dp_in = out_5
    
    for i in range(2):
            
        dp_in = DprnnBlock_stateful(numUnits = 128, batch_size = 1, L = -1,width = 50 ,channel = 128, causal=True)(dp_in)
            
    dp_out = dp_in

    name = 'decoder'
    skipcon_1 = Concatenate(axis = -1)([out_5,dp_out])
    
    deconv_1 = DeConv2D_stateful(filters = DeCNN_filter_list[0], keranel_size = (2,3), strides = (1,1), padding = 'same', state_shape = DeCNN_state_list[0],name = name+'_dconv_1')(skipcon_1)
    dbn_1 = BatchNormalization(name = name+'_dbn_1')(deconv_1)
    dout_1 = PReLU(shared_axes=[1,2])(dbn_1)
    
    skipcon_2 = Concatenate(axis = -1)([out_4,dout_1])
    
    deconv_2 = DeConv2D_stateful(filters = DeCNN_filter_list[1], keranel_size = (2,3), strides = (1,1), padding = 'same', state_shape = DeCNN_state_list[1],name = name+'_dconv_2')(skipcon_2)
    dbn_2 = BatchNormalization(name = name+'_dbn_2')(deconv_2)
    dout_2 = PReLU(shared_axes=[1,2])(dbn_2)
    
    skipcon_3 = Concatenate(axis = -1)([out_3,dout_2])
    
    deconv_3 = DeConv2D_stateful(filters = DeCNN_filter_list[2], keranel_size = (2,3), strides = (1,1), padding = 'same', state_shape = DeCNN_state_list[2],name = name+'_dconv_3')(skipcon_3)
    dbn_3 = BatchNormalization(name = name+'_dbn_3')(deconv_3)
    dout_3 = PReLU(shared_axes=[1,2])(dbn_3)
    
    skipcon_4 = Concatenate(axis = -1)([out_2,dout_3])
    
    deconv_4 = DeConv2D_stateful(filters = DeCNN_filter_list[3], keranel_size = (2,3), strides = (1,2), padding = 'same', state_shape = DeCNN_state_list[3],name = name+'_dconv_4')(skipcon_4)
    dbn_4 = BatchNormalization(name = name+'_dbn_4')(deconv_4)
    dout_4 = PReLU(shared_axes=[1,2])(dbn_4)
    
    skipcon_5 = Concatenate(axis = -1)([out_1,dout_4])
    
    deconv_5 = DeConv2D_stateful(filters = DeCNN_filter_list[4], keranel_size = (2,5), strides = (1,2), padding = 'valid', state_shape = DeCNN_state_list[4],name = name+'_dconv_5')(skipcon_5)
    
    output_mask = deconv_5[:,1:-1,:-2]
    
    enh_spec = Lambda(mk_mask)([real,imag,output_mask])
        
    #enh_real, enh_imag = enh_spec[0],enh_spec[1]
        
    enh_frame = Lambda(ifftLayer,arguments = {'mode':'real_imag'})(enh_spec)[:,0,:]
    enh_frame = enh_frame * win
    
    model = Model([Encoder_input],
                    [enh_frame])
    
    model_encoder = Model(Encoder_input,dp_in)
    
    upop = tf.get_collection('upop')
    rsop = tf.get_collection('rsop')
    session_list = [[Encoder_input], [enh_frame,upop]]
    #model_encoder.load_weights('./real_time_model_encoder.h5')
    return model_encoder,model,session_list,rsop

#%%
def test(f,encoder_not_RT):
    
    s = librosa.load(f,16000)[0]
    l = len(s)
    
    L = (l - blockLen)//blockshift + 1
    
    n_samples = (L-1) * blockshift + blockLen
    input_spec_data = s[:n_samples]
    input_spec_data = np.expand_dims(input_spec_data,axis=0)
    
    begin = time.time()
    Not_RT_output = encoder_not_RT.predict(input_spec_data)
    end = time.time()
    print('not RT {} s\n'.format(end - begin))
    
    RT_output = []
    begin = time.time()
    encoder_RT.reset_states()
    for i in range(L):
        
        output_data = sess.run(session_list[1],feed_dict = {session_list[0][0]:input_spec_data[:,i*blockshift:i*blockshift+blockLen]})[0]#encoder_RT.predict(input_spec_data[:,i*blockshift:i*blockshift+blockLen])[0]
    
        RT_output.append(output_data)
        
    end = time.time()
    print('RT {} per frames\n'.format((end - begin)/L))
    
    RT_output = np.concatenate(RT_output,axis = 0)  
    RT_output = overlapadd(RT_output)
    
    plt.subplot(311)
    plt.plot(s)
    plt.subplot(312)
    plt.plot(Not_RT_output[0])
    plt.title('not real time')
    plt.subplot(313)
    plt.plot(RT_output)
    plt.title('real time')
    return RT_output,Not_RT_output

def test_RT(f,model,sess,reset = False,plot = False):
    '''
    f: wav file to be enhanced
    model: stateful model
    sess: tensorflow session of keras backend
    reset: if True, reset all the buffers, including hidden states of LSTM and paddings of convolutional layers
    '''
    if reset:
        sess.run(reset_op)
        
    s = sf.read(f,dtype = 'float32')[0]
    l = len(s)
    
    L = (l - blockLen)//blockshift + 1
    
    n_samples = (L-1) * blockshift + blockLen
    input_spec_data = s[:n_samples]
    input_spec_data = np.expand_dims(input_spec_data,axis=0)
    
    RT_output = []
    begin = time.time()

    for i in range(L):
        
        output_data = sess.run(session_list[1],feed_dict = {model.input:input_spec_data[:,i*blockshift:i*blockshift+blockLen]})[0]#encoder_RT.predict(input_spec_data[:,i*blockshift:i*blockshift+blockLen])[0]
    
        RT_output.append(output_data)
        
    end = time.time()
    print('Real time processing: total {} s for {} s audio\n'.format((end - begin),l/16000))
    print('Real time processing: {} s per frame\n'.format((end - begin)/L))
    
    RT_output = np.concatenate(RT_output,axis = 0)  
    RT_output = overlapadd(RT_output)
    if plot:
        plt.subplot(211)
        plt.plot(s)
        plt.title('noisy speech')
        plt.subplot(212)
        plt.plot(RT_output)
        plt.title('enhanced speech by real-time inference')
    sf.write('./enhance_s.wav',RT_output.astype(np.float32),16000)

    return RT_output

if __name__ == '__main__':
    
    _,encoder_RT,session_list,reset_op = mk_Encoder_RT()
    sess = K.get_session()
    sess.run(tf.global_variables_initializer())
  
    encoder_RT.load_weights('../pretrain_model/model_DPCRN_SNR+logMSE_causal_sinw.h5')
    # warm-up
    test_RT('../test.wav',model = encoder_RT,sess = sess,reset =True,plot = False)
    #test
    test_RT('../test.wav',model = encoder_RT,sess = sess,reset =True,plot = True)
    
    #print the computational complexity (floating operations)
    graph =tf.get_default_graph()
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    print('Total Float Opreators per frame:',flops.total_float_ops)
    # MACC is approximately equal to FLOPs / 2
    print('Total Float Opreators per second:',flops.total_float_ops * 16000 / 200)