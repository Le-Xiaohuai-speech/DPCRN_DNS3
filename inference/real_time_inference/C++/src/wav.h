/* Provides basic handling of PCM format RIFF/WAVE audio files
   See
   http://www-mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/Docs/riffmci.pdf
   for some information on the relevant specification.
   This is not as elaborate as the WAVE file support in the "audiofile" package
   but is very simple, self-contained, and probably adequate for most purposes.
*/

#include <stdint.h>
#include <stdio.h>

// wav data
struct wav_data{
	public:
		int16_t* data;
		int size;

		wav_data(){
			data=NULL;
			size=0;
		}	
};
// wav header info 
struct wav_info {
   uint_fast16_t num_channels;      /* 1 for mono, 2 for stereo, etc. */
   uint_fast16_t bits_per_sample;   /* 16 for CD, 24 for high-res, etc. */
   uint_fast32_t sample_rate;       /* 44100 for CD, 88200, 96000, 192000, etc. */
   uint_fast32_t num_samples;       /* total number of samples per channel */
};

void read_wav_info(struct wav_info* w, wav_data* wdata, FILE* fp);
/* Read wav_info from *fp, assuming *fp is a PCM format RIFF/WAVE file.
   Leaves the seek position of *fp at the beginning of the data section
   of *fp, so one could immediately begin reading/writing samples */

void write_wav_hdr(const struct wav_info* w, FILE* fp);
/* Write a standard 44-byte PCM format RIFF/WAVE header to the beginning of *fp.
   Again, the seek position of *fp will be left at the beginning of
   the data section, so one can immediately begin writing samples */

void print_wav_info(const struct wav_info* w);
/* Prints information from *w to stdout */

void write_file_wav(const struct wav_info* w, FILE* fp, const int16_t* sample);

void free_source(wav_data* data);

// read pcm data as .bin or .pcm
void read_file_bin_data(const char *file, void *data, size_t byte_length);

// write pcm data as .bin or .pcm
void write_file_bin_data(const char *file, void *data, size_t byte_length);
