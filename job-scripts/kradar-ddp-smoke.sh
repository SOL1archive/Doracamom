#!/usr/bin/env bash
#SBATCH --job-name=kradar-ddp-smoke
#SBATCH --partition=suma_a6000,gigabyte_a6000,tyan_a6000
#SBATCH --qos=big_qos
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --time=00:25:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

# Smoke test for the K-Radar DDP fix (find_unused_parameters=True).
# Reuses the already-converted toy split in data/KRadar (8 train / 2 val).
# Runs a single epoch + one eval on 2 GPUs with per-iteration logging so we can
# confirm training gets past iteration 0 (the original run hung there) and that
# the BEV dimension chain does not crash. Tight --time so a hang fails fast.

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/lustre/thesol1/Doracamom}"
CONFIG="${CONFIG:-projects/configs/Doracamom/Doracamom_KRadar.py}"
WORK_DIR="${WORK_DIR:-work_dirs/Doracamom_KRadar/smoke_${SLURM_JOB_ID}}"
GPUS="${GPUS:-2}"
PORT="${PORT:-28531}"

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
# Surface NCCL errors instead of silently hanging until the 30-min watchdog.
export NCCL_ASYNC_ERROR_HANDLING=1
export PORT

mkdir -p "${MPLCONFIGDIR}" logs/slurm "${WORK_DIR}"

missing=0
for path in "${CONFIG}" data/KRadar/kradar_infos_train.pkl data/KRadar/kradar_infos_val.pkl; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing required input: ${path}" >&2
    missing=1
  fi
done
if [[ "${missing}" -ne 0 ]]; then
  echo "Re-run the K-Radar converter before this smoke test." >&2
  exit 2
fi

echo "Smoke test started on $(hostname) at $(date)"
echo "Config:   ${CONFIG}"
echo "Work dir: ${WORK_DIR}"
echo "GPUs:     ${GPUS}"
module list
nvidia-smi
python -c "import torch, mmcv, mmdet, mmdet3d; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available())"

# Per-iteration logging, 1 epoch only. dist_train.sh appends --deterministic.
./tools/dist_train.sh \
  "${CONFIG}" \
  "${GPUS}" \
  --work-dir "${WORK_DIR}" \
  --no-validate \
  --cfg-options \
    log_config.interval=1 \
    runner.max_epochs=1 \
    total_epochs=1 \
    checkpoint_config.interval=1

echo "Smoke test finished at $(date)"
echo "If you see iter [1..4] loss lines above, DDP no longer hangs."
