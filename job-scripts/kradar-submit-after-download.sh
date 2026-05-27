#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/lustre/thesol1/Doracamom}"
RAW_DATA_ROOT="${RAW_DATA_ROOT:-/scratch2/thesol1/radar-dataset/K-Radar}"
SEQUENCE_ID="${SEQUENCE_ID:-58}"
EXPECTED_BYTES="${EXPECTED_BYTES:-225753822303}"
CHECK_INTERVAL="${CHECK_INTERVAL:-300}"
ZIP_PATH="${RAW_DATA_ROOT}/${SEQUENCE_ID}.zip"

cd "${REPO_ROOT}"
mkdir -p logs/login logs/slurm

echo "Submit watcher started on $(hostname) at $(date)"
echo "Watching ${ZIP_PATH} until it reaches ${EXPECTED_BYTES} bytes"

while true; do
  current_size=0
  if [[ -f "${ZIP_PATH}" ]]; then
    current_size="$(stat -c%s "${ZIP_PATH}")"
  fi
  part_size=0
  if [[ -f "${ZIP_PATH}.part" ]]; then
    part_size="$(stat -c%s "${ZIP_PATH}.part")"
  fi
  echo "$(date): zip=${current_size} part=${part_size}"
  if [[ "${current_size}" -ge "${EXPECTED_BYTES}" && ! -f "${ZIP_PATH}.part" ]]; then
    break
  fi
  sleep "${CHECK_INTERVAL}"
done

echo "Download appears complete at $(date)"
ls -lh "${ZIP_PATH}"
file "${ZIP_PATH}"

unpack_job="$(sbatch --parsable job-scripts/kradar-sample-unpack.sbatch)"
echo "Submitted unpack job: ${unpack_job}"

train_job="$(
  sbatch --parsable \
    --dependency=afterok:${unpack_job} \
    --export=ALL,SAMPLE_TRAIN_FRAMES=8,SAMPLE_VAL_FRAMES=2,PROFILE_SAMPLES=10,PROFILE_WARMUP=3 \
    job-scripts/doracamom-kradar-a6000-train-eval-profile.sbatch
)"
echo "Submitted train/eval/profile job: ${train_job}"

for _ in $(seq 1 120); do
  squeue -j "${unpack_job},${train_job}" -o '%.18i %.9P %.24j %.8T %.10M %.20R' || true
  sleep 60
done
