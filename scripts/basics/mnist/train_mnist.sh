#!/bin/bash
# sh scripts/basics/mnist/train_mnist.sh

GPU=1
CPU=1
node=73
jobname=openood

PYTHONPATH='.':$PYTHONPATH \

python main.py \
--config configs/datasets/mnist/mnist.yml \
configs/preprocessors/base_preprocessor.yml \
configs/networks/lenet.yml \
configs/pipelines/train/baseline.yml
