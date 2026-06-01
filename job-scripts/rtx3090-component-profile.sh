#!/usr/bin/env bash
#SBATCH --job-name=doracamom-component-profile
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

# Per-component FLOPs/latency profiling. Writes a detailed markdown + JSON
# report under profiles/<config-stem>_<timestamp>/.

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/lustre/thesol1/Doracamom}"
CONFIG="${CONFIG:-projects/configs/Doracamom/Doracamom_TJ4D.py}"
CHECKPOINT="${CHECKPOINT:-ckpts/doracamom_tj4d.pth}"
SAMPLES="${SAMPLES:-50}"
WARMUP="${WARMUP:-20}"
PORT="${PORT:-28541}"

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

mkdir -p "${MPLCONFIGDIR}" logs/slurm profiles

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
module list
nvidia-smi
python -c "import torch; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available())"

python tools/analysis_tools/profile_components.py \
  "${CONFIG}" \
  "${CHECKPOINT}" \
  --samples "${SAMPLES}" \
  --warmup "${WARMUP}"

echo "Job finished at $(date)"
