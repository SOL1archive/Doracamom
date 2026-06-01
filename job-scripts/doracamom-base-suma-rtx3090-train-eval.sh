#!/usr/bin/env bash
#SBATCH --job-name=doracamom-train-eval
#SBATCH --partition=dell_rtx3090
#SBATCH --qos=big_qos
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -Eeuo pipefail

REPO_ROOT="/lustre/thesol1/Doracamom"
CONFIG="${CONFIG:-projects/configs/Doracamom/Doracamom.py}"
GPUS="${GPUS:-2}"
WORK_DIR="${WORK_DIR:-work_dirs/doracamom_20241120/final/Doracamom_1120_final}"
FINAL_EPOCH="${FINAL_EPOCH:-16}"
PORT="${PORT:-28509}"

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

required_files=(
  "data/NewScenes_Final/newscenes-final_infos_temporal_occ_train.pkl"
  "data/NewScenes_Final/newscenes-final_infos_temporal_occ_val.pkl"
  "ckpts/r50_fcos3d_pretrain.pth"
  "ckpts/radarpillarnet.pth"
)

missing=0
for path in "${required_files[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    missing=1
  fi
done

if [[ "${missing}" -ne 0 ]]; then
  echo "Place the missing dataset metadata and pretrained checkpoints, then resubmit this job." >&2
  exit 2
fi

echo "Job started on $(hostname) at $(date)"
echo "Config: ${CONFIG}"
echo "GPUs: ${GPUS}"
echo "Work dir: ${WORK_DIR}"
module list
nvidia-smi
python -c "import torch, mmcv, mmdet, mmdet3d, detectron2, torch_scatter; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available()); print('mmcv', mmcv.__version__); print('mmdet', mmdet.__version__); print('mmdet3d', mmdet3d.__version__); print('detectron2', detectron2.__version__); print('torch_scatter', torch_scatter.__version__)"

echo "Starting training at $(date)"
./tools/dist_train.sh "${CONFIG}" "${GPUS}" --work-dir "${WORK_DIR}"

final_ckpt="${WORK_DIR}/epoch_${FINAL_EPOCH}.pth"
if [[ ! -f "${final_ckpt}" && -f "${WORK_DIR}/latest.pth" ]]; then
  final_ckpt="${WORK_DIR}/latest.pth"
fi

if [[ ! -f "${final_ckpt}" ]]; then
  echo "Training finished, but no checkpoint was found at epoch_${FINAL_EPOCH}.pth or latest.pth in ${WORK_DIR}" >&2
  exit 3
fi

echo "Starting eval with ${final_ckpt} at $(date)"
./tools/dist_test.sh "${CONFIG}" "${final_ckpt}" "${GPUS}"

echo "Job finished at $(date)"
