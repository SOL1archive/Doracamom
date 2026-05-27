#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29503}

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

PYTHONPATH="${ROOT}:${ROOT}/deform_attn_3d":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    "${ROOT}/tools/test.py" $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --eval bbox
