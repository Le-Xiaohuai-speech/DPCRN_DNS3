#include <iostream>
#include <cstring>
#include "wav.h"
#include "fft.h"
/*
run g++ ./test.cpp ./wav.cpp ./fft.cpp -o read_wav
*/

int main(int, char**) {
    std::cout << "input the wav file" << std::endl;
    std::string wav_file;
    // file_name = "./sample.wav"; 
    std::cin >> wav_file;
    FILE* fp = fopen(wav_file.data(), "rb");
    // header information and sample data
    wav_info info;
    wav_data wdata;
    // read the wav file
    read_wav_info(&info, &wdata, fp);
    fclose(fp);   
    
    float len_in_s = (float)info.num_samples / (float)info.sample_rate;
    // print the information
    std::cout << "file name: " << wav_file << "\n"
              << "channel number: " << info.num_channels <<"\n"
              << "bits per sampe: "  << info.bits_per_sample << "\n"
              << "sample rate: " << info.sample_rate<<"\n"
              << "num samples: " << info.num_samples<<"\n"
              << "time in second: " << len_in_s << std::endl; 

   std::cout << "samples: "<< std::endl;

   for(int i=0;i<100;i++){
      std::cout << wdata.data[i]<<" ";
   }
    std::cout << std::endl;

    // test STFT
    int fft_len = 512;
    int hop_len = 256;
    int N_frame = wdata.size / hop_len + 1;
    float win[512];
    float *s, *spec;
    std::string win_f = "../bin/win.bin";
    s = new float[wdata.size];
    spec = new float[N_frame * fft_len * 2];
    std::cout << "frame length: " << fft_len << "\n"
              << "hop length: " << hop_len << "\n"
              << "frame number: " << N_frame << std::endl;
    // read window
    read_file_bin_data(win_f.c_str(), win, 512*4);
    // STFT
    for (int i=0; i < wdata.size; i++){
	s[i] = float(wdata.data[i]) / 32767.0;
    }
    STFT(s, spec, win, wdata.size, fft_len, fft_len, hop_len);
    /*
    check the output in python by:
    x = np.fromfile('../bin/stft.bin', dtype=np.float32) 
    */
    write_file_bin_data("../bin/stft.bin", spec, N_frame * fft_len * 2 * 4);
    // iSTFT
    memset(s, 0, wdata.size*4);
    iSTFT(s, spec, win, N_frame, wdata.size, fft_len, fft_len, hop_len);

    for (int i=0; i < wdata.size; i++){
	wdata.data[i] = int16_t(32767.0 * s[i]);
    }
    // write the signal as a .pcm file
    write_file_bin_data("./output_s.pcm", wdata.data, wdata.size * info.bits_per_sample / 8);
    // write the signal as a .wav file
    fp = fopen("./output_s.wav", "wb");
    write_file_wav(&info, fp, wdata.data);
    fclose(fp);  

    free_source(&wdata);
    delete [] s;
    delete [] spec;
}
