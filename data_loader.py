# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 14:57:00 2021

@author: xiaohuaile
"""
import soundfile as sf
#from wavinfo import WavInfoReader
from random import shuffle, seed
import numpy as np
import librosa
import os
from scipy import signal
'''
TRAIN_DIR: DNS data
RIR_DIR: Room impulse response
'''
TRAIN_DIR = '/data/ssd1/xiaohuai.le/DNS_data1/DNS_data'
RIR_DIR = '/data/ssd1/xiaohuai.le/RIR_database/impulse_responses/'


#FIR, frequencies below 60Hz will be filtered
fir = signal.firls(1025,[0,40,50,60,70,8000],[0,0,0.1,0.5,1,1],fs = 16000)

def add_pyreverb(clean_speech, rir):
    '''
    convolve RIRs to the clean speech to generate reverbrant speech
    '''
    l = len(rir)//2
    reverb_speech = signal.fftconvolve(clean_speech, rir, mode="full")
    # make reverb_speech same length as clean_speech
    reverb_speech = reverb_speech[l : clean_speech.shape[0]+l]

    return reverb_speech
#按照snr混合音频
def mk_mixture(s1,s2,snr,eps = 1e-8):
    '''
    make mixture from s1 and s2 with snr
    '''
    norm_sig1 = s1 / np.sqrt(np.sum(s1 ** 2) + eps) 
    norm_sig2 = s2 / np.sqrt(np.sum(s2 ** 2) + eps)
    alpha = 10**(snr/20)
    mix = norm_sig2 + alpha*norm_sig1
    M = max(np.max(abs(mix)),np.max(abs(norm_sig2)),np.max(abs(alpha*norm_sig1))) + eps
    mix = mix / M
    norm_sig1 = norm_sig1 * alpha/ M
    norm_sig2 = norm_sig2 / M

    return norm_sig1,norm_sig2,mix,snr


class data_generator():
    
    def __init__(self,train_dir = TRAIN_DIR, 
                    RIR_dir = RIR_DIR,
                    validation_rate=0.1,
                    length_per_sample = 4,
                    fs = 16000,
                    n_fft = 400,
                    n_hop = 200,
                    batch_size = 8,
                    sample_num=-1, 
                    add_reverb = True,
                    reverb_rate = 0.5
                    ):
        '''
        keras data generator
        Para.:
            train_dir:  folder storing training data, including train_dir/clean, train_dir/noise
            RIR_dir:    folder storing RIRs, from OpenSLR26 and OpenSLR28
            validation_rate: how much data is used for validation
            length_per_sample: speech sample length in second
            fs: sample rate of the speech
            n_fft: FFT length and window length in STFT
            n_hop: hop length in STFT
            batch_size: batch size
            sample_num: how many samples are used for training and validation
            add_reverb: adding reverbrantion or not
            reverb_rate: how much data is reverbrant
        '''
        
        self.train_dir = train_dir
        self.clean_dir = os.path.join(train_dir,'clean')
        self.noise_dir = os.path.join(train_dir,'noise')
        
        self.fs = fs
        self.batch_size = batch_size 
        self.length_per_sample = length_per_sample 
        self.L = length_per_sample * self.fs
        # calculate the length of each sample after iSTFT
        self.points_per_sample = ((self.L - n_fft) // n_hop) * n_hop + n_fft
        
        self.validation_rate = validation_rate
        self.add_reverb = add_reverb
        self.reverb_rate = reverb_rate
        
        if RIR_dir is not None:
            self.rir_dir = RIR_dir
            self.rir_list = librosa.util.find_files(self.rir_dir,ext = 'wav')[:sample_num]
            np.random.shuffle(self.rir_list)
            self.rir_list = self.rir_list[:sample_num]
            print('there are {} rir clips\n'.format(len(self.rir_list)))

        self.noise_file_list = os.listdir(self.noise_dir)
        self.clean_file_list = os.listdir(self.clean_dir)[:sample_num]
        self.train_length = int(len(self.clean_file_list)*(1-validation_rate))
        self.train_list, self.validation_list = self.generating_train_validation(self.train_length)
        self.valid_length = len(self.validation_list)
        
        self.train_rir = self.rir_list[:self.train_length]
        self.valid_rir = self.rir_list[self.train_length : self.train_length + self.valid_length]
        print('have been generated DNS training list...\n')
       
        print('there are {} samples for training, {} for validation'.format(self.train_length,self.valid_length))

    def find_files(self,file_name):
        '''
        from file_name find parallel noise file and noisy file 
        e.g.
        file_name: clean_fileid_1.wav
        noise_file_name: noise_fileid_1.wav
        noisy_file_name: noisy_fileid_1.wav
        '''
        #noise_file_name = np.random.choice(self.noise_file_list) #randomly selection
        _,k1,k2 = file_name.split('_')
        noise_file_name = 'noise' + '_' + k1 + '_' + k2
        noisy_file_name = 'noisy' + '_' + k1 + '_' + k2
        
        # random segmentation
        Begin_S = int(np.random.uniform(0,30 - self.length_per_sample)) * self.fs
        Begin_N = int(np.random.uniform(0,30 - self.length_per_sample)) * self.fs
        return noise_file_name,noisy_file_name,Begin_S,Begin_N
     
    def generating_train_validation(self,training_length):
        '''
        get training and validation data
        '''
        np.random.shuffle(self.clean_file_list)
        self.train_list,self.validation_list = self.clean_file_list[:training_length],self.clean_file_list[training_length:]

        return self.train_list,self.validation_list
      
    def generator(self, batch_size, validation = False):
        '''
        data generator,
            validation: if True, get validation data genertor
        '''
        if validation:
            train_data = self.validation_list
            train_rir = self.valid_rir
        else:
            train_data = self.train_list
            train_rir = self.train_rir
        N_batch = len(train_data) // batch_size
        batch_num = 0
        while (True):

            batch_clean = np.zeros([batch_size,self.points_per_sample],dtype = np.float32)
            batch_noisy = np.zeros([batch_size,self.points_per_sample],dtype = np.float32)
            
            for i in range(batch_size):
                # random amplitude gain
                gain = np.random.normal(loc=-5,scale=10)
                gain = 10**(gain/10)
                gain = min(gain,3)
                gain = max(gain,0.01)
                
                SNR = np.random.uniform(-5,5)
                sample_num = batch_num*batch_size + i
                #get the path of clean audio
                clean_f = train_data[sample_num]
                rir_f = train_rir[sample_num]
                reverb_rate = np.random.rand()

                noise_f, noisy_f, Begin_S,Begin_N = self.find_files(clean_f)
                clean_s = sf.read(os.path.join(self.clean_dir,clean_f),dtype = 'float32',start= Begin_S,stop = Begin_S + self.points_per_sample)[0]
                noise_s = sf.read(os.path.join(self.noise_dir,noise_f),dtype = 'float32',start= Begin_N,stop = Begin_N + self.points_per_sample)[0]

                clean_s = add_pyreverb(clean_s, fir)
                
                #noise_s = noise_s - np.mean(noise_s)
                if self.add_reverb:
                    if reverb_rate < self.reverb_rate:
                        rir_s = sf.read(rir_f,dtype = 'float32')[0]
                        if len(rir_s.shape)>1:
                            rir_s = rir_s[:,0]
                        clean_s = add_pyreverb(clean_s, rir_s)
                        
                clean_s,noise_s,noisy_s,_ = mk_mixture(clean_s,noise_s,SNR,eps = 1e-8)

                batch_clean[i,:] = clean_s * gain
                batch_noisy[i,:] = noisy_s * gain

            batch_num += 1

            if batch_num == N_batch:
                batch_num = 0

                if validation:
                    train_data = self.validation_list
                    train_rir = self.valid_rir
                else:
                    train_data = self.train_list
                    train_rir = self.train_rir

                np.random.shuffle(train_data)
                np.random.shuffle(train_rir)
                np.random.shuffle(self.noise_file_list)

                N_batch = len(train_data) // batch_size

            yield batch_noisy,batch_clean
            

