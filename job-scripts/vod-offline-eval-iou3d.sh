#!/usr/bin/env bash
#SBATCH --job-name=doracamom-vod-eval-iou3d
#SBATCH --partition=dell_rtx3090
#SBATCH --qos=big_qos
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/lustre/thesol1/Doracamom}"
CONFIG="${CONFIG:-projects/configs/Doracamom/Doracamom_vod.py}"
PRED_PKL="${PRED_PKL:-test/Doracamom_vod/Tue_Jun__2_21_13_06_2026pts_bbox.pkl}"
WORK_DIR="${WORK_DIR:-work_dirs/rtx3090_basic_metrics/vod_eval_iou3d_${SLURM_JOB_ID}}"
export CONFIG PRED_PKL WORK_DIR

cd "${REPO_ROOT}"

source /etc/profile.d/modules.sh 2>/dev/null || true
module unload cuda/12.8 cuda/12.5 cuda/12.1 cuda/11.8 2>/dev/null || true
module load cuda/11.8

source /home/thesol1/miniconda3/etc/profile.d/conda.sh
conda activate doracamom

export CUDA_HOME=/opt/ohpc/pub/apps/cuda/11.8
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/deform_attn_3d:${PYTHONPATH:-}"
export MPLCONFIGDIR="/tmp/matplotlib-${USER}-${SLURM_JOB_ID}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TORCH_CUDA_ARCH_LIST="8.6"

mkdir -p "${MPLCONFIGDIR}" logs/slurm "${WORK_DIR}"

echo "Job started on $(hostname) at $(date)"
echo "Config: ${CONFIG}"
echo "Prediction pkl: ${PRED_PKL}"
echo "Work dir: ${WORK_DIR}"
module list
nvidia-smi

python - <<'PY' | tee "${WORK_DIR}/offline_kitti_eval_iou3d.log"
import importlib
import os
import time

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmdet3d.datasets import build_dataset
from mmdet3d.ops.iou3d import iou3d_cuda
from mmdet3d.core.evaluation import kitti_eval
from mmdet3d.core.evaluation.kitti_utils import eval as eval_mod


def xywhr_to_xyxyr_np(boxes):
    out = np.zeros_like(boxes, dtype=np.float32)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    out[:, 4] = boxes[:, 4]
    return out


def bev_overlap_iou3d(boxes, qboxes, criterion=-1):
    boxes = np.asarray(boxes, dtype=np.float32)
    qboxes = np.asarray(qboxes, dtype=np.float32)
    if boxes.shape[0] == 0 or qboxes.shape[0] == 0:
        return np.zeros((boxes.shape[0], qboxes.shape[0]), dtype=np.float32)
    a_np = xywhr_to_xyxyr_np(boxes)
    b_np = xywhr_to_xyxyr_np(qboxes)
    a = torch.from_numpy(a_np).cuda()
    b = torch.from_numpy(b_np).cuda()
    out = a.new_zeros((a.shape[0], b.shape[0]))
    if criterion == -1:
        iou3d_cuda.boxes_iou_bev_gpu(a.contiguous(), b.contiguous(), out)
    else:
        iou3d_cuda.boxes_overlap_bev_gpu(a.contiguous(), b.contiguous(), out)
        area_a = torch.from_numpy((boxes[:, 2] * boxes[:, 3])).cuda().view(-1, 1)
        area_b = torch.from_numpy((qboxes[:, 2] * qboxes[:, 3])).cuda().view(1, -1)
        if criterion == 0:
            out = out / torch.clamp(area_a, min=1e-8)
        elif criterion == 1:
            out = out / torch.clamp(area_b, min=1e-8)
        elif criterion == 2:
            pass
        else:
            out = out / torch.clamp(area_a + area_b - out, min=1e-8)
    return out.detach().cpu().numpy().astype(np.float32)


def d3_overlap_iou3d(boxes, qboxes, criterion=-1):
    rinc = bev_overlap_iou3d(
        boxes[:, [0, 2, 3, 5, 6]],
        qboxes[:, [0, 2, 3, 5, 6]],
        criterion=2)
    eval_mod.d3_box_overlap_kernel(boxes, qboxes, rinc, criterion)
    return rinc


eval_mod.bev_box_overlap = bev_overlap_iou3d
eval_mod.d3_box_overlap = d3_overlap_iou3d

cfg = Config.fromfile(os.environ["CONFIG"])
importlib.import_module("projects.mmdet3d_plugin")
cfg.data.test.test_mode = True
dataset = build_dataset(cfg.data.test)
dt_annos = mmcv.load(os.environ["PRED_PKL"])
gt_annos = [info["annos"] for info in dataset.data_infos]
print("dataset", len(dataset), "detections", len(dt_annos), "classes", dataset.CLASSES, flush=True)
t0 = time.time()
result_str, ap_dict = kitti_eval(
    gt_annos, dt_annos, dataset.CLASSES, eval_types=["bbox", "bev", "3d"])
print("elapsed_s", time.time() - t0)
print(result_str)
print("EVAL_DICT", ap_dict)
mmcv.dump({"result_str": result_str, "ap_dict": ap_dict},
          os.path.join(os.environ["WORK_DIR"], "offline_kitti_eval_iou3d.json"))
PY

echo "Job finished at $(date)"
