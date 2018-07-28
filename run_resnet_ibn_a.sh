#!/usr/bin/env sh
export MXNET_CUDNN_AUTOTUNE_DEFAULT=1

## train resnet_ibn-50 
python -u train_resnet_ibn_a.py --data-dir data/imagenet --data-type imagenet --depth 50 --batch-size 256 --gpus=0,1,2,3
