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
  
    // write the signal as a .pcm file
    write_file_bin_data("./output_s.pcm", wdata.data, wdata.size * info.bits_per_sample / 8);
    // write the signal as a .wav file
    fp = fopen("./output_s.wav", "wb");
    write_file_wav(&info, fp, wdata.data);
    fclose(fp);  

    free_source(&wdata);
}
