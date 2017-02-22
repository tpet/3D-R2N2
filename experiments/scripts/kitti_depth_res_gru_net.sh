#!/bin/bash

set -x
set -e

export CUDA_VISIBLE_DEVICES=
export PYTHONUNBUFFERED="True"

NET_NAME=KittiDepthResGRUNet
EXP_DETAIL=default_model
OUT_PATH='./output/'$NET_NAME/$EXP_DETAIL
LOG="$OUT_PATH/log.`date +'%Y-%m-%d_%H-%M-%S'`"
# CFG=experiments/cfgs/kitti_depth.yaml
CFG=experiments/cfgs/kitti_depth_small.yaml

# Make the dir if it not there
mkdir -p $OUT_PATH
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# export THEANO_FLAGS="floatX=float32,device=gpu,assert_no_cpu_op='raise'"
export THEANO_FLAGS="floatX=float32,device=cpu,assert_no_cpu_op='raise'"
# Debug flags
# export THEANO_FLAGS="$THEANO_FLAGS,optimizer=fast_compile"
# export THEANO_FLAGS="$THEANO_FLAGS,optimizer=None"
# export THEANO_FLAGS="$THEANO_FLAGS,exception_verbosity=high"

# --batch-size 24 \
python main.py \
      --batch-size 8 \
      --iter 60000 \
      --out $OUT_PATH \
      --model $NET_NAME \
      --cfg $CFG \
      ${*:1}

echo 'Training completed. Running test net...'

python main.py \
      --test \
      --batch-size 1 \
      --out $OUT_PATH \
      --weights $OUT_PATH/weights.npy \
      --model $NET_NAME \
      --cfg $CFG \
      ${*:1}

