# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 22:16:58 2021

@author: xiaohuai.le
"""
import numpy as np
import tflite_runtime.interpreter as tflite
import matplotlib.pyplot as plt
import time
import copy
import soundfile as sf
import librosa

def enhancement_stateful(noisy_f, model_stateful = './dpcrn_stateful_model.tflite', output_f = './enhance_s.wav', plot = True, gain =1):

    noisy_s = sf.read(noisy_f,dtype = 'float32')[0] * gain

    length = len(noisy_s)
    
    N_frame = (length - 512) // 256 + 1 
    
    enh_s = np.zeros([512 + 256 * (N_frame - 1)],dtype = np.float32)
    
    inp = np.zeros([1,1,257,3], dtype = np.float32)
    inp_state_1 = np.zeros([1,32,128], dtype = np.float32)
    inp_state_2 = np.zeros([1,32,128], dtype = np.float32)
    t = []

    win = np.sin(np.arange(.5,512-.5+1)/512*np.pi) 
    
    interpreter = tflite.Interpreter(model_path = model_stateful)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    [print(i['name'],i['shape']) for i in input_details]
    [print(i['name'],i['shape']) for i in output_details]
    for i in range(N_frame):
        begin = time.perf_counter()
        
        noisy = noisy_s[i*256 : i*256 + 512] * win
        spec = np.fft.rfft(noisy).astype('complex64')
        spec1 = copy.copy(spec)
        
        inp[0,0,:,0] = spec1.real
        inp[0,0,:,1] = spec1.imag
        inp[0,0,:,2] = 2 * np.log(abs(spec))
        
        interpreter.set_tensor(input_details[0]['index'], inp)
        interpreter.set_tensor(input_details[1]['index'], inp_state_1)
        interpreter.set_tensor(input_details[2]['index'], inp_state_2)
        
        interpreter.invoke()
        
        mag_mask = interpreter.get_tensor(output_details[0]['index'])[0,0]
        sin = interpreter.get_tensor(output_details[1]['index'])[0,0]
        cos = interpreter.get_tensor(output_details[2]['index'])[0,0]
        inp_state_1 = interpreter.get_tensor(output_details[3]['index'])
        inp_state_2 = interpreter.get_tensor(output_details[4]['index'])
        
        spec = spec * mag_mask * (cos + 1j*sin)
        
        enhanced = np.fft.irfft(spec) * win
        
        end = time.perf_counter()
        enh_s[i*256 : i*256 + 512] += enhanced
        t.append(end-begin)
    
    print('Total {} frames, inference time per frame:{}s'.format(N_frame,np.mean(t)))
    if plot:
       
        spec_n = librosa.stft(noisy_s,400,200,center = False)
        spec_e = librosa.stft(enh_s, 400,200,center = False)
        plt.figure(0)
        plt.plot(noisy_s)
        plt.plot(enh_s)
        plt.figure(1)
        plt.subplot(211)
        plt.imshow(np.log(abs(spec_n)),cmap= 'jet',origin ='lower')
        plt.subplot(212)
        plt.imshow(np.log(abs(spec_e)),cmap= 'jet',origin ='lower')

    sf.write(output_f,enh_s,16000)
    return noisy_s,enh_s

if __name__ == '__main__':
    
    
    n,e = enhancement_stateful('D:/codes/test_audio/librispeech/white0/61-70968-0030.wav',model_stateful = './dpcrn_stateful_model.tflite')
