#!/usr/bin/env bash
#SBATCH --job-name=kradar-sample-unpack
#SBATCH --partition=dell_cpu
#SBATCH --qos=cpu_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -Eeuo pipefail

RAW_DATA_ROOT="${RAW_DATA_ROOT:-/scratch2/thesol1/radar-dataset/K-Radar}"
SEQUENCE_ID="${SEQUENCE_ID:-58}"
ZIP_PATH="${RAW_DATA_ROOT}/${SEQUENCE_ID}.zip"

mkdir -p logs/slurm

echo "Unpack job started on $(hostname) at $(date)"
echo "Zip path: ${ZIP_PATH}"
df -h "${RAW_DATA_ROOT}"

if [[ -d "${RAW_DATA_ROOT}/${SEQUENCE_ID}/info_label" ]]; then
  echo "Sequence ${SEQUENCE_ID} already appears unpacked."
  exit 0
fi

if [[ ! -f "${ZIP_PATH}" ]]; then
  echo "Missing zip: ${ZIP_PATH}" >&2
  exit 2
fi

echo "Zip info:"
ls -lh "${ZIP_PATH}"
file "${ZIP_PATH}"

echo "Unpacking at $(date)"
unzip -q -n "${ZIP_PATH}" -d "${RAW_DATA_ROOT}"

if ! find "${RAW_DATA_ROOT}" -mindepth 1 -maxdepth 4 -type d -name info_label -print -quit | grep -q .; then
  echo "Unpack completed, but no info_label directory was found." >&2
  exit 3
fi

echo "Available K-Radar sequence roots:"
find "${RAW_DATA_ROOT}" -mindepth 1 -maxdepth 3 -type d -name info_label -print | sed 's#/info_label$##' | head -20
echo "Unpack job finished at $(date)"
