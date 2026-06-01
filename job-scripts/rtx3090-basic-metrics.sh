#!/usr/bin/env bash
#SBATCH --job-name=doracamom-rtx3090-metrics
#SBATCH --partition=dell_rtx3090
#SBATCH --qos=big_qos
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/lustre/thesol1/Doracamom}"
CONFIG="${CONFIG:-projects/configs/Doracamom/Doracamom_vod.py}"
CHECKPOINT="${CHECKPOINT:-ckpts/doracamom_vod.pth}"
RUN_NAME="${RUN_NAME:-$(basename "${CONFIG}" .py)}"
WORK_DIR="${WORK_DIR:-work_dirs/rtx3090_basic_metrics/${RUN_NAME}_${SLURM_JOB_ID}}"
GPUS="${GPUS:-1}"
PORT="${PORT:-28521}"
PROFILE_SAMPLES="${PROFILE_SAMPLES:-100}"
PROFILE_WARMUP="${PROFILE_WARMUP:-100}"
STDOUT_LOG="${STDOUT_LOG:-logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out}"

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

mkdir -p "${MPLCONFIGDIR}" logs/slurm ckpts "${WORK_DIR}"

missing=0
for path in "${CONFIG}" "${CHECKPOINT}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    missing=1
  fi
done
if [[ "${missing}" -ne 0 ]]; then
  echo "Place the checkpoint under ckpts/ or override CHECKPOINT, then resubmit." >&2
  exit 2
fi

echo "Job started on $(hostname) at $(date)"
echo "Config: ${CONFIG}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Work dir: ${WORK_DIR}"
echo "GPUs: ${GPUS}"
module list
nvidia-smi
python -c "import torch, mmcv, mmdet, mmdet3d; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available()); print('mmcv', mmcv.__version__); print('mmdet', mmdet.__version__); print('mmdet3d', mmdet3d.__version__)"

echo "Running bbox evaluation at $(date)"
./tools/dist_test.sh \
  "${CONFIG}" \
  "${CHECKPOINT}" \
  "${GPUS}"

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
  --stdout-log "${STDOUT_LOG}" \
  --profile-json "${WORK_DIR}/latency_flops.json" \
  --output-md "${WORK_DIR}/basic_metrics_report.md" \
  --output-json "${WORK_DIR}/basic_metrics_report.json"

echo "Results:"
echo "  latency/FLOPs: ${WORK_DIR}/latency_flops.json"
echo "  report markdown: ${WORK_DIR}/basic_metrics_report.md"
echo "  report json: ${WORK_DIR}/basic_metrics_report.json"
echo "Job finished at $(date)"
