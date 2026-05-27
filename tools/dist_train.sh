#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-28509}

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

PYTHONPATH="${ROOT}:${ROOT}/deform_attn_3d":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    "${ROOT}/tools/train.py" $CONFIG --launcher pytorch ${@:3} --deterministic
