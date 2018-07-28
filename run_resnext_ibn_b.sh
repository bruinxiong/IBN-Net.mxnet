#!/usr/bin/env sh
export MXNET_CUDNN_AUTOTUNE_DEFAULT=1

# you'd better change setting with your own --data-dir, --depth, --batch-size, --gpus.

## train resnext-50
python -u train_resnext_ibn_b.py --data-dir data/imagenet --data-type imagenet --depth 50 --batch-size 128 --num-group 32 --gpus=0,1,2,3
