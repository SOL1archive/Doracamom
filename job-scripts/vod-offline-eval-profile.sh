#!/usr/bin/env bash
#SBATCH --job-name=doracamom-vod-offline-eval
#SBATCH --partition=dell_rtx3090
#SBATCH --qos=big_qos
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/lustre/thesol1/Doracamom}"
CONFIG="${CONFIG:-projects/configs/Doracamom/Doracamom_vod.py}"
CHECKPOINT="${CHECKPOINT:-ckpts/doracamom_vod.pth}"
PRED_PKL="${PRED_PKL:-test/Doracamom_vod/Tue_Jun__2_21_13_06_2026pts_bbox.pkl}"
WORK_DIR="${WORK_DIR:-work_dirs/rtx3090_basic_metrics/vod_val_w100_${SLURM_JOB_ID}}"
PROFILE_SAMPLES="${PROFILE_SAMPLES:-100}"
PROFILE_WARMUP="${PROFILE_WARMUP:-100}"
PORT="${PORT:-28561}"
export CONFIG CHECKPOINT PRED_PKL WORK_DIR

cd "${REPO_ROOT}"

source /etc/profile.d/modules.sh 2>/dev/null || true
module unload cuda/12.8 cuda/12.5 cuda/12.1 cuda/11.8 2>/dev/null || true
module load cuda/11.8

source /home/thesol1/miniconda3/etc/profile.d/conda.sh
conda activate doracamom

export CUDA_HOME=/opt/ohpc/pub/apps/cuda/11.8
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/nvvm/lib64:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/deform_attn_3d:${PYTHONPATH:-}"
export MPLCONFIGDIR="/tmp/matplotlib-${USER}-${SLURM_JOB_ID}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TORCH_CUDA_ARCH_LIST="8.6"
export PORT

mkdir -p "${MPLCONFIGDIR}" logs/slurm "${WORK_DIR}"

echo "Job started on $(hostname) at $(date)"
echo "Config: ${CONFIG}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Prediction pkl: ${PRED_PKL}"
echo "Work dir: ${WORK_DIR}"
module list
nvidia-smi

echo "Running offline KITTI eval at $(date)"
python - <<'PY' | tee "${WORK_DIR}/offline_kitti_eval.log"
import importlib
import os
import mmcv
from mmcv import Config
from mmdet3d.datasets import build_dataset
from mmdet3d.core.evaluation import kitti_eval

cfg = Config.fromfile(os.environ["CONFIG"])
importlib.import_module("projects.mmdet3d_plugin")
cfg.data.test.test_mode = True
dataset = build_dataset(cfg.data.test)
dt_annos = mmcv.load(os.environ["PRED_PKL"])
gt_annos = [info["annos"] for info in dataset.data_infos]
print("dataset", len(dataset), "detections", len(dt_annos), "classes", dataset.CLASSES)
result_str, ap_dict = kitti_eval(
    gt_annos, dt_annos, dataset.CLASSES, eval_types=["bbox", "bev", "3d"])
print(result_str)
print("EVAL_DICT", ap_dict)
mmcv.dump({"result_str": result_str, "ap_dict": ap_dict},
          os.path.join(os.environ["WORK_DIR"], "offline_kitti_eval.json"))
PY

echo "Profiling latency and FLOPs at $(date)"
python tools/analysis_tools/profile_latency_flops.py \
  "${CONFIG}" \
  "${CHECKPOINT}" \
  --samples "${PROFILE_SAMPLES}" \
  --warmup "${PROFILE_WARMUP}" \
  --output "${WORK_DIR}/latency_flops.json"

echo "Writing automated report at $(date)"
python tools/analysis_tools/make_basic_metrics_report.py \
  --work-dir "${WORK_DIR}" \
  --stdout-log "${WORK_DIR}/offline_kitti_eval.log" \
  --profile-json "${WORK_DIR}/latency_flops.json" \
  --output-md "${WORK_DIR}/basic_metrics_report.md" \
  --output-json "${WORK_DIR}/basic_metrics_report.json"

echo "Results:"
echo "  offline eval: ${WORK_DIR}/offline_kitti_eval.log"
echo "  latency/FLOPs: ${WORK_DIR}/latency_flops.json"
echo "  report markdown: ${WORK_DIR}/basic_metrics_report.md"
echo "  report json: ${WORK_DIR}/basic_metrics_report.json"
echo "Job finished at $(date)"
