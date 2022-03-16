# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 16:05:31 2021

@author: xiaohuai.le
"""

import pyaudio
import tkinter as tk
import wave
import threading
import queue
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.lines as line
import numpy as np
from soundfile import write
import tflite_runtime.interpreter as tflite

#%%
interpreter = tflite.Interpreter(model_path = './dpcrn_stateful_model.tflite')
interpreter.allocate_tensors()
    
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
    
[print('input_{}'.format(index),i['shape']) for index, i in enumerate(output_details)]
[print('output_{}'.format(index),i['shape']) for index,i in enumerate(output_details)]
#%%
CHUNK = 256
N_FFT = 512
hop = 256
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

data =[]
frames=[]
counter=1
N = 200
window = np.sin(np.arange(.5,N_FFT-.5+1)/N_FFT*np.pi) 
gain = 1
MAX = 32767/gain
frame = np.zeros(N_FFT) #256*4

noisy_s = []
enh_s = []

#GUI
class Application(tk.Frame):
    def __init__(self,master=None):
        tk.Frame.__init__(self,master)
        self.grid()
        self.creatWidgets()

    def creatWidgets(self):
        self.quitButton=tk.Button(self,text='quit',command=root.destroy)
        self.quitButton.grid(column=1,row=3)


#make noisy axes and enhance axes
fig = plt.figure()
noisy_ax = plt.subplot(325,xlim=(0,CHUNK*N), ylim=(-MAX,MAX))
enhance_ax = plt.subplot(326,xlim=(0,CHUNK*N), ylim=(-MAX,MAX))
noisy_ax.set_title("noisy signal")
enhance_ax.set_title("enhanced signal")
noisy_line = line.Line2D([],[])
enhance_line = line.Line2D([],[])
#plot data update after reading buffer
noisy_data = np.zeros(CHUNK*N,dtype=np.int16)
enhance_data = np.zeros(CHUNK*N,dtype=np.int16)
noisy_x_data = np.arange(0,CHUNK*N,1)
enhance_x_data = np.arange(0,CHUNK*N,1)

n_stft_ax = plt.subplot(311)
n_stft_ax.set_title("noisy spectrogram")
n_stft_ax.set_ylim(0,N_FFT//2 + 1)
n_stft_ax.set_xlim(0,1000)
n_image_stft = n_stft_ax.imshow(np.random.randn(N_FFT//2 + 1,1000),cmap ='jet')
n_stft_data=np.zeros([257,1000],dtype=np.float32)

stft_ax = plt.subplot(312)
stft_ax.set_title("enhanced spectrogram")
stft_ax.set_ylim(0,N_FFT//2 + 1)
stft_ax.set_xlim(0,1000)
image_stft = stft_ax.imshow(np.random.randn(N_FFT//2 + 1,1000),cmap ='jet')
stft_data=np.zeros([N_FFT//2 + 1,1000],dtype=np.float32)

def plot_init():
    noisy_ax.add_line(noisy_line)
    enhance_ax.add_line(enhance_line)
    return enhance_line,noisy_line,image_stft,n_image_stft 
    
def plot_update(i):
    global noisy_data
    global enhance_data
    global enhance_x_data
    global stft_data
    global n_stft_data
    noisy_line.set_xdata(noisy_x_data)
    noisy_line.set_ydata(noisy_data)
    
    enhance_line.set_xdata(enhance_x_data)
    enhance_line.set_ydata(enhance_data)
    
    image_stft.set_data(stft_data)
    n_image_stft.set_data(n_stft_data)
    return enhance_line,noisy_line,image_stft,n_image_stft 


def audio_callback(in_data, frame_count, time_info, status):
    global ad_rdy_ev

    q.put(in_data)
    ad_rdy_ev.set()
    if counter <= 0:
        return (None,pyaudio.paComplete)
    else:
        return (None,pyaudio.paContinue)

#processing block

def read_audio_thead(q,stream,frames,ad_rdy_ev):
    global frame 
    inp = np.zeros([1,1,257,3], dtype = np.float32)
    inp_state_1 = np.zeros([1,32,128], dtype = np.float32)
    inp_state_2 = np.zeros([1,32,128], dtype = np.float32)
    while stream.is_active():
        ad_rdy_ev.wait(timeout=1000)
        if not q.empty():
            #process audio data here
            data=q.get()
            while not q.empty():
                q.get()
            # CHUNK * N_chunk
            noisy_data_0 = np.frombuffer(data,np.dtype('<i2'))
            noisy_s.append(noisy_data_0)
            
            noisy_data[:CHUNK*(N-1)] = noisy_data[CHUNK:CHUNK*N]
            noisy_data[CHUNK*(N-1):] = noisy_data_0
            
            noisy_frame = noisy_data[-N_FFT:] / 32768.0 
            noisy_frame = noisy_frame * window
            noisy_fft = np.fft.rfft(noisy_frame)
            
            inp[0,0,:,0] = noisy_fft.real
            inp[0,0,:,1] = noisy_fft.imag
            inp[0,0,:,2] = 2 * np.log(abs(noisy_fft))
          
            n_stft_data[:,:-1] = n_stft_data[:,1:]
            n_stft_data[:,-1] = inp[0,0,:,2]    

            #%%
            interpreter.set_tensor(input_details[0]['index'], inp)
            interpreter.set_tensor(input_details[1]['index'], inp_state_1)
            interpreter.set_tensor(input_details[2]['index'], inp_state_2)
        
            interpreter.invoke()
        
            mag_mask = interpreter.get_tensor(output_details[0]['index'])[0,0]
            sin = interpreter.get_tensor(output_details[1]['index'])[0,0]
            cos = interpreter.get_tensor(output_details[2]['index'])[0,0]
            inp_state_1 = interpreter.get_tensor(output_details[3]['index'])
            inp_state_2 = interpreter.get_tensor(output_details[4]['index'])
        
            noisy_fft = noisy_fft * mag_mask * (cos + 1j*sin)
        
            enhance_frame = np.fft.irfft(noisy_fft) * window
                
            frame[CHUNK:] = frame[CHUNK:] + enhance_frame[:CHUNK]
            
            enh_s.append((frame[CHUNK:] * 32768).astype(np.int16))
            
            stft_data[:,:-1] = stft_data[:,1:]
            stft_data[:,-1] = np.log(abs(np.fft.rfft(frame * window))**2 + 1e-9)      
            
            frame[:CHUNK] = frame[CHUNK:]
            frame[CHUNK:] = enhance_frame[CHUNK:]
            
            enhance_data[:-CHUNK] = enhance_data[CHUNK:]
            enhance_data[-CHUNK:] = ( frame[:CHUNK] * 32768).astype(np.int16)
            
        ad_rdy_ev.clear()
        

ani = animation.FuncAnimation(fig, plot_update,
                              init_func=plot_init, 
                              frames=1,
                              interval=30,
                              blit=True)


# pyaudio
p = pyaudio.PyAudio()
q = queue.Queue()
stream = p.open(format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        output=False,
        frames_per_buffer=CHUNK,
        stream_callback=audio_callback)

print("Start Recording")
stream.start_stream()
ad_rdy_ev=threading.Event()

t=threading.Thread(target=read_audio_thead,args=(q,stream,frames,ad_rdy_ev))
t.daemon=True
t.start()
plt.show()

n = np.concatenate(noisy_s)
s = np.concatenate(enh_s)
write('./noise_s.wav',n,16000)
write('./enh_s.wav',s,16000)

print("Recorded")

