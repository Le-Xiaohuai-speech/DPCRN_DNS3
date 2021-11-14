# DPCRN_DNS3
*Created on Mon Oct 28 16:05:31 2021* </br>
*@author: xiaohuai.le*

This repository is the official implementation of paper "DPCRN: Dual-Path Convolution Recurrent Network for Single Channel Speech Enhancement". This work got the third place in Deep Noise Suppression Challenge.
## Requirements
tensorflow>=1.14, </br>
numpy,            </br>
matplotlib,       </br>
librosa,          </br>
sondfile.         </br>

## Datasets
 We use [Deep Noise Suppression Dataset](https://github.com/microsoft/DNS-Challenge) and [OpenSLR26](http://www.openslr.org/26/), [OpenSLR28](http://www.openslr.org/28/) RIRs dataset in our training and validation stages. The directory structure of the dataset is shown below: </br>
dataset </br>
├── clean    </br>
│    ├── audio1.wav</br>
│    ├── audio2.wav</br>
│    ├── audio3.wav</br>
│    ...</br>
├── noise</br>
│    ├── audio1.wav</br>
│    ├── audio2.wav</br>
│    ├── audio3.wav</br>
│    ...</br>

RIR</br>
├── rirs</br>
│    ├── rir1.wav</br>
│    ├── rir2.wav</br>
│    ├── rir3.wav</br>
│    ...</br>

## Training and test
Run the following code to training:
```shell
python main.py --mode train --cuda 0 --experimentName experiment_1
```
Run the following code to test the model on a single file:
```shell
python main.py --mode test --test_dir the_dir_of_noisy --output_dir the_dir_of_enhancement_results
```
## More samples 

The final results on the blind test set of DNS3 is available on https://github.com/Le-Xiaohuai-speech/DPCRN_DNS3_Results. </br>

## Real-time inference
**Note that the real-time inference can only run on the tensorflow=1.x.**
Run real-time inference to calculate the time cost of a frame:</br>  
```shell
python ./real_time_processing/real_time_DPCRN.py
```
## Tensorflow Lite quantization and pruning

##Citations
```shell
@inproceedings{le21b_interspeech,
  author={Xiaohuai Le and Hongsheng Chen and Kai Chen and Jing Lu},
  title={{DPCRN: Dual-Path Convolution Recurrent Network for Single Channel Speech Enhancement}},
  year=2021,
  booktitle={Proc. Interspeech 2021},
  pages={2811--2815},
  doi={10.21437/Interspeech.2021-296}
}
```
