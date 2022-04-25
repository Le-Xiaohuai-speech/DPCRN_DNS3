# Real Time Inference
*Created on Mon Apr 25 17:40:30 2022* </br>
*@author: xiaohuai.le*

A smaller DPCRN model is used for real time inference with about 0.53 M parameters and 1.1 G MACs. The LSTMs of the baseline model are replaced by GRUs and the frame size and hop size are set to 32 ms and 16 ms respectively.
