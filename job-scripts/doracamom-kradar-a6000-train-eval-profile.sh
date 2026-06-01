#!/usr/bin/env bash
#SBATCH --job-name=doracamom-kradar
#SBATCH --partition=suma_a6000,gigabyte_a6000,tyan_a6000
#SBATCH --qos=big_qos
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --time=14-00:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/lustre/thesol1/Doracamom}"
RAW_DATA_ROOT="${RAW_DATA_ROOT:-/scratch2/thesol1/radar-dataset/K-Radar}"
DATA_ROOT="${DATA_ROOT:-data/KRadar}"
CONFIG="${CONFIG:-projects/configs/Doracamom/Doracamom_KRadar.py}"
WORK_DIR="${WORK_DIR:-work_dirs/Doracamom_KRadar/a6000_${SLURM_JOB_ID}}"
GPUS="${GPUS:-2}"
FINAL_EPOCH="${FINAL_EPOCH:-16}"
PORT="${PORT:-28509}"
PROFILE_SAMPLES="${PROFILE_SAMPLES:-200}"
PROFILE_WARMUP="${PROFILE_WARMUP:-20}"
SAMPLE_TRAIN_FRAMES="${SAMPLE_TRAIN_FRAMES:-8}"
SAMPLE_VAL_FRAMES="${SAMPLE_VAL_FRAMES:-2}"
TRAIN_SPLIT="${TRAIN_SPLIT:-}"
VAL_SPLIT="${VAL_SPLIT:-}"

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

mkdir -p "${MPLCONFIGDIR}" logs/slurm "${WORK_DIR}" "${DATA_ROOT}"

echo "Job started on $(hostname) at $(date)"
echo "Raw K-Radar root: ${RAW_DATA_ROOT}"
echo "Converted data root: ${DATA_ROOT}"
echo "Config: ${CONFIG}"
echo "Work dir: ${WORK_DIR}"
module list
nvidia-smi
python -c "import torch, mmcv, mmdet, mmdet3d; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available()); print('mmcv', mmcv.__version__); print('mmdet', mmdet.__version__); print('mmdet3d', mmdet3d.__version__)"

if ! find "${RAW_DATA_ROOT}" -mindepth 1 -maxdepth 4 -type d -name info_label -print -quit | grep -q .; then
  echo "No K-Radar sequence with info_label was found under ${RAW_DATA_ROOT}." >&2
  echo "Download or unpack the raw K-Radar sequences there before resubmitting." >&2
  exit 2
fi

if [[ ! -f "${DATA_ROOT}/kradar_infos_train.pkl" || ! -f "${DATA_ROOT}/kradar_infos_val.pkl" ]]; then
  split_args=()
  if [[ -z "${TRAIN_SPLIT}" && -z "${VAL_SPLIT}" && "${SAMPLE_TRAIN_FRAMES}" -gt 0 ]]; then
    TRAIN_SPLIT="${DATA_ROOT}/kradar_sample_train_split.txt"
    VAL_SPLIT="${DATA_ROOT}/kradar_sample_val_split.txt"
    total_frames=$((SAMPLE_TRAIN_FRAMES + SAMPLE_VAL_FRAMES))
    mapfile -t label_paths < <(find "${RAW_DATA_ROOT}" -path "*/info_label/*.txt" -type f | sort | head -n "${total_frames}")
    : > "${TRAIN_SPLIT}"
    : > "${VAL_SPLIT}"
    for idx in "${!label_paths[@]}"; do
      label_path="${label_paths[$idx]}"
      seq="$(basename "$(dirname "$(dirname "${label_path}")")")"
      frame="$(basename "${label_path}" .txt)"
      if [[ "${idx}" -lt "${SAMPLE_TRAIN_FRAMES}" ]]; then
        printf "%s,%s\n" "${seq}" "${frame}" >> "${TRAIN_SPLIT}"
      else
        printf "%s,%s\n" "${seq}" "${frame}" >> "${VAL_SPLIT}"
      fi
    done
    echo "Generated sampled splits: train=$(wc -l < "${TRAIN_SPLIT}") val=$(wc -l < "${VAL_SPLIT}")"
  fi
  if [[ -n "${TRAIN_SPLIT}" ]]; then
    split_args+=(--train-split "${TRAIN_SPLIT}")
  fi
  if [[ -n "${VAL_SPLIT}" ]]; then
    split_args+=(--val-split "${VAL_SPLIT}")
  fi
  echo "Creating K-Radar infos at $(date)"
  python tools/create_data.py kradar \
    --root-path "${RAW_DATA_ROOT}" \
    --out-dir "${DATA_ROOT}" \
    --val-ratio 0.2 \
    --cube-percentile 99.0 \
    --max-points 20000 \
    "${split_args[@]}"
else
  echo "Found existing K-Radar infos; skipping conversion."
fi

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

echo "Starting evaluation with ${final_ckpt} at $(date)"
./tools/dist_test.sh "${CONFIG}" "${final_ckpt}" "${GPUS}"

echo "Profiling latency and FLOPs at $(date)"
python tools/analysis_tools/profile_latency_flops.py \
  "${CONFIG}" \
  "${final_ckpt}" \
  --samples "${PROFILE_SAMPLES}" \
  --warmup "${PROFILE_WARMUP}" \
  --output "${WORK_DIR}/profile_latency_flops.json"

echo "Profile written to ${WORK_DIR}/profile_latency_flops.json"
echo "Job finished at $(date)"
