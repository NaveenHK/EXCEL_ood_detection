#!/bin/bash

# GPU=1
# CPU=1
# node=73
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
# srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \

python main.py \
    --config configs/datasets/imagenet200/imagenet200.yml \
    configs/datasets/imagenet200/imagenet200_ood.yml \
    configs/networks/resnet18_224x224.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/excel.yml \
    --num_workers 8 \
    --network.checkpoint 'results/imagenet200_resnet18_224x224_base_e90_lr0.1_default/s0/best.ckpt' \
    --mark 1 \

#./results/imagenet200_resnet18_224x224_base_e90_lr0.1_default
#    --postprocessor.postprocessor_args.pcacomponents 8
#'./results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt' \