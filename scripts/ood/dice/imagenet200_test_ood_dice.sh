#!/bin/bash
# sh scripts/ood/dice/imagenet200_test_ood_dice.sh

############################################
# alternatively, we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/eval_ood.py
# especially if you want to get results from
# multiple runs

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
    configs/postprocessors/dice.yml \
    --num_workers 8 \
    --network.checkpoint 'results/imagenet200_resnet18_224x224_base_e90_lr0.1_default/s0/best.ckpt' \
    --mark 1 \

## ood
#python scripts/eval_ood.py \
#    --id-data imagenet200 \
#    --root ./results/imagenet200_resnet18_224x224_base_e90_lr0.1_default \
#    --postprocessor dice \
#    --save-score --save-csv #--fsood
#
## full-spectrum ood
#python scripts/eval_ood.py \
#    --id-data imagenet200 \
#    --root ./results/imagenet200_resnet18_224x224_base_e90_lr0.1_default \
#    --postprocessor dice \
#    --save-score --save-csv --fsood
