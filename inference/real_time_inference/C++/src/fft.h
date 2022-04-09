/* @breif Fast Fourier Transform
 * @Author: Xiaohuaile
 * @Date: 2022-4-9
 * @Last Modified by: Xiaohuaile
 */
#ifndef FFT_H_
#define FFT_H_
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <cstdio>

#define M_PI 3.141592653589793238462643383279502

static void make_sintbl(int n, float* sintbl);

static void make_bitrev(int n, int* bitrev);

/*
* @breif Fast Fourier Transform
* @x: real part
* @y: imaginary part
* @n: length of fft, negative for inverse FFT
*/
int fft(float* x, float* y, int n);

/*
* @brief: Short Time Fourier Transform
* @s: input signal
* @spec: spectrogram, N_frame * N_fft *2 (real and imag)
* @win: window, window_length
* @signal_length
* @fft_length
* @window_length
* @hop_length
*/
void STFT(float *s, float *spec, float *win, int signal_length, int fft_length, int window_length, int hop_length);
 
/*
* @brief: inverse Short Time Fourier Transform
* @s: output signal
* @spec: spectrogram, N_frame * N_fft / 2 + 1 (rfft part) + N_fft/2 - 1 (rfft symmetrical part) 
* @win: window, window_length
* @N_frame: number of the frame
* @signal_length: length of the input signal
* @fft_length
* @window_length
* @hop_length
*/
void iSTFT(float *s, float *spec, float *win, int N_frame, int signal_length, int fft_length, int window_length, int hop_length);

#endif


