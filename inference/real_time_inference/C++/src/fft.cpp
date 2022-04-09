#include "fft.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <cstdio>

static void make_sintbl(int n, float* sintbl)
{
	int i, n2, n4, n8;
	float c, s, dc, ds, t;

	n2 = n / 2;  n4 = n / 4;  n8 = n / 8;
	t = sin(M_PI / n);
	dc = 2 * t * t;  ds = sqrt(dc * (2 - dc));
	t = 2 * dc;  c = sintbl[n4] = 1;  s = sintbl[0] = 0;
	for (i = 1; i < n8; i++) {
		c -= dc;  dc += t * c;
		s += ds;  ds -= t * s;
		sintbl[i] = s;  sintbl[n4 - i] = c;
	}
	if (n8 != 0) sintbl[n8] = sqrt(0.5);
	for (i = 0; i < n4; i++)
		sintbl[n2 - i] = sintbl[i];
	for (i = 0; i < n2 + n4; i++)
		sintbl[i + n2] = -sintbl[i];
}

static void make_bitrev(int n, int* bitrev)
{
	int i, j, k, n2;

	n2 = n / 2;  i = j = 0;
	for (;;) {
		bitrev[i] = j;
		if (++i >= n) break;
		k = n2;
		while (k <= j) { j -= k; k /= 2; }
		j += k;
	}
}

int fft(float* x, float* y, int n)
{
	static int    last_n = 0;     /* previous n */
	static int    *bitrev = NULL; /* bit reversal table */
	static float  *sintbl = NULL; /* trigonometric function table */
	int i, j, k, ik, h, d, k2, n4, inverse;
	float t, s, c, dx, dy;

	/* preparation */
	if (n < 0) {
		n = -n;  inverse = 1; /* inverse transform */
	}
	else {
		inverse = 0;
	}
	n4 = n / 4;
	if (n != last_n || n == 0) {
		last_n = n;
		if (sintbl != NULL) free(sintbl);
		if (bitrev != NULL) free(bitrev);
		if (n == 0) return 0;             /* free the memory */
		sintbl = (float*)malloc((n + n4) * sizeof(float));
		bitrev = (int*)malloc(n * sizeof(int));
		if (sintbl == NULL || bitrev == NULL) {
			//Error("%s in %f(%d): out of memory\n", __func__, __FILE__, __LINE__);
			return 1;
		}
		make_sintbl(n, sintbl);
		make_bitrev(n, bitrev);
	}

	/* bit reversal */
	for (i = 0; i < n; i++) {
		j = bitrev[i];
		if (i < j) {
			t = x[i];  x[i] = x[j];  x[j] = t;
			t = y[i];  y[i] = y[j];  y[j] = t;
		}
	}

	/* transformation */
	for (k = 1; k < n; k = k2) {
		h = 0;  k2 = k + k;  d = n / k2;
		for (j = 0; j < k; j++) {
			c = sintbl[h + n4];
			if (inverse)
				s = -sintbl[h];
			else
				s = sintbl[h];
			for (i = j; i < n; i += k2) {
				ik = i + k;
				dx = s * y[ik] + c * x[ik];
				dy = c * y[ik] - s * x[ik];
				x[ik] = x[i] - dx;  x[i] += dx;
				y[ik] = y[i] - dy;  y[i] += dy;
			}
			h += d;
		}
	}
	if (inverse) {
		/* divide by n in case of the inverse transformation */
		for (i = 0; i < n; i++)
		{
			x[i] /= n;
			y[i] /= n;
		}
	}
	return 0;  /* finished successfully */
}

void STFT(float *s, float *spec, float *win, int signal_length, int fft_length, int window_length, int hop_length){

    int N_frame = signal_length / hop_length + 1;
    int index = 0;
   
    for (int i = 0; i < N_frame; i++){
        for (int j = 0; j < window_length; j++ ){
            index = i * hop_length + j;
            if (index < signal_length){
                spec[i * fft_length * 2 + j] = win[j] * s[index]; 
            } 
        }
        fft(spec + i * fft_length * 2, spec + i * fft_length * 2 + fft_length, fft_length);
    }
}

void iSTFT(float *s, float *spec, float *win, int N_frame, int signal_length, int fft_length, int window_length, int hop_length){

    for (int i = 0; i < N_frame; i++){
        for (int j = 1; j < fft_length / 2; j++ ){
			// real part
			spec[i * fft_length * 2 + fft_length - j] = spec[i * fft_length * 2 + j];
			// imaginary part
			spec[i * fft_length * 2 + 2 * fft_length - j] = -spec[i * fft_length * 2 + fft_length + j];
        }
        fft(spec + i * fft_length * 2, spec + i * fft_length * 2 + fft_length, -fft_length);
		// add window // overlap-add
		for (int k = 0; k < fft_length ; k++ ){
			//spec[i * fft_length * 2 + k] = win[k] * spec[i * fft_length * 2 + k]
			if (i * hop_length + k < signal_length){
				s[i * hop_length + k] = s[i * hop_length + k] + win[k] * spec[i * fft_length * 2 + k];
			}
		}
    }
}


