/* Provides basic handling of PCM format RIFF/WAVE audio files
   See
   http://www-mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/Docs/riffmci.pdf
   for some information on the relevant specification.
   This is not as elaborate as the WAVE file support in the "audiofile" package
   but is very simple, self-contained, and probably adequate for most purposes.
*/

#include <stdint.h>
#include <stdio.h>
  
struct wav_info {
   uint_fast16_t num_channels;      /* 1 for mono, 2 for stereo, etc. */
   uint_fast16_t bits_per_sample;   /* 16 for CD, 24 for high-res, etc. */
   uint_fast32_t sample_rate;       /* 44100 for CD, 88200, 96000, 192000, etc. */
   uint_fast32_t num_samples;       /* total number of samples per channel */
};

void read_wav_info(struct wav_info* w, FILE* fp);
/* Read wav_info from *fp, assuming *fp is a PCM format RIFF/WAVE file.
   Leaves the seek position of *fp at the beginning of the data section
   of *fp, so one could immediately begin reading/writing samples */

void write_wav_hdr(const struct wav_info* w, FILE* fp);
/* Write a standard 44-byte PCM format RIFF/WAVE header to the beginning of *fp.
   Again, the seek position of *fp will be left at the beginning of
   the data section, so one can immediately begin writing samples */

void print_wav_info(const struct wav_info* w);
/* Prints information from *w to stdout */

void write_sample(const struct wav_info* w, FILE* fp, const int_fast32_t* sample);
/* Write a sample to *fp in the correct Little Endian format.
   sample should be an array with w->num_channels elements.
   Note that we use the int_fast32_t datatype to hold samples, which should be
   changed if you want to use bits_per_sample > 32.  Also, if you use, say,
   bits_per_sample=24, then you want to make sure that your actual samples
   are going to fit into a 3-byte Little Endian integer in twos-complement
   encoding.  If you're only going to use bits_per_sample=16, you could
   use int_fast16_t instead of int_fast32_t.  Note also that the WAVE
   file format expects 8-bit samples to be UNsigned, so if you're going
   to use bits_per_sample=8, then you should use uint_fast8_t to hold
   your samples. */
