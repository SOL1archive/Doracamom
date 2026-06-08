#!/usr/bin/env bash
#SBATCH --job-name=doracamom-vod-profile
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
WORK_DIR="${WORK_DIR:-work_dirs/rtx3090_basic_metrics/vod_profile_${SLURM_JOB_ID}}"
PROFILE_SAMPLES="${PROFILE_SAMPLES:-100}"
PROFILE_WARMUP="${PROFILE_WARMUP:-100}"
PORT="${PORT:-28571}"

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
export PORT

mkdir -p "${MPLCONFIGDIR}" logs/slurm "${WORK_DIR}"

echo "Job started on $(hostname) at $(date)"
echo "Config: ${CONFIG}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Work dir: ${WORK_DIR}"
module list
nvidia-smi

python tools/analysis_tools/profile_latency_flops.py \
  "${CONFIG}" \
  "${CHECKPOINT}" \
  --samples "${PROFILE_SAMPLES}" \
  --warmup "${PROFILE_WARMUP}" \
  --output "${WORK_DIR}/latency_flops.json"

echo "Job finished at $(date)"
