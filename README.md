# ST-GRU
Here provides the entire code of paper "Motion Prediction for Beating Heart Surgery with GRU"

## Citation

## Setup
Experiments are run with Pytorch 1.12.0, CUDA 11.3, Python 3.7.13.

## Code and datasets
We provide ``` DNNprediction ''' for prediction of future steps on S-DNN model,
``` GRUprediction ''' for predicting results on T-GRU and ST-GRU models.
''' cycleprediction ''' is the implementation of the MSE evalution based on cyclical pattern.

750 frame stereo images are provided in '''measuresPhantom.mat''' and '''measuresInvivo.mat'''.
Frame 1 to 600 from both in vivo and phantom dataset are used for training and frame 601 to 750 for testing.
