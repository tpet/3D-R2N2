#!/bin/bash

set -x
set -e

# export CUDA_VISIBLE_DEVICES=
export PYTHONUNBUFFERED="True"

NET_NAME=KittiDepthResGRUNet
EXP_DETAIL=kitti_full_depth_128
OUT_PATH='./output/'$NET_NAME/$EXP_DETAIL
LOG="$OUT_PATH/log.`date +'%Y-%m-%d_%H-%M-%S'`"

CFG_TRN=experiments/cfgs/kitti_full_depth_128_trnval.yaml
CFG_TST=experiments/cfgs/kitti_full_depth_128_tst.yaml

# Make the dir if it not there
mkdir -p $OUT_PATH
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

export THEANO_FLAGS="floatX=float32,device=gpu,assert_no_cpu_op='raise'"
# export THEANO_FLAGS="floatX=float32,device=cpu,assert_no_cpu_op='raise'"

python main.py \
      --batch-size 1 \
      --iter 25600 \
      --out $OUT_PATH \
      --model $NET_NAME \
      --cfg $CFG_TRN \
      ${*:1}

echo 'Training completed. Running test net...'

python main.py \
      --test \
      --batch-size 1 \
      --out $OUT_PATH \
      --weights $OUT_PATH/weights.npy \
      --model $NET_NAME \
      --cfg $CFG_TST \
      ${*:1}

