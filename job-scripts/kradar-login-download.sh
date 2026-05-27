#!/usr/bin/env bash
set -Eeuo pipefail

RAW_DATA_ROOT="${RAW_DATA_ROOT:-/scratch2/thesol1/radar-dataset/K-Radar}"
SEQUENCE_ID="${SEQUENCE_ID:-58}"
FILE_ID="${FILE_ID:-11a5pGnJpl4fhgFiVPXP9jwjEiUoqm2Eb}"
CHUNK_BYTES="${CHUNK_BYTES:-1073741824}"
ZIP_PATH="${RAW_DATA_ROOT}/${SEQUENCE_ID}.zip"

mkdir -p "${RAW_DATA_ROOT}"

echo "Login-node download started on $(hostname) at $(date)"
echo "Zip path: ${ZIP_PATH}"
df -h "${RAW_DATA_ROOT}"

probe="$(mktemp /tmp/kradar-gdrive-login-XXXXXX.html)"
wget --no-check-certificate -O "${probe}" \
  "https://drive.google.com/uc?export=download&id=${FILE_ID}"

uuid="$(sed -n 's/.*name="uuid" value="\([^"]*\)".*/\1/p' "${probe}")"
if [[ -z "${uuid}" ]]; then
  echo "Could not find Google Drive confirmation uuid in ${probe}" >&2
  exit 2
fi

url="https://drive.usercontent.google.com/download?id=${FILE_ID}&export=download&confirm=t&uuid=${uuid}"
current_size=0
if [[ -f "${ZIP_PATH}" ]]; then
  current_size="$(stat -c%s "${ZIP_PATH}")"
fi

header="$(mktemp /tmp/kradar-gdrive-header-XXXXXX.txt)"
curl -L -sS -I -r "${current_size}-$((current_size + 1023))" \
  "${url}" > "${header}"
if ! grep -qi '^HTTP/.* 206' "${header}"; then
  echo "Google Drive did not accept byte-range resume. Header follows:" >&2
  cat "${header}" >&2
  exit 3
fi

total_size="$(tr -d '\r' < "${header}" | sed -n 's#^content-range: bytes [0-9]*-[0-9]*/\([0-9]*\).*#\1#Ip' | tail -1)"
if [[ -z "${total_size}" ]]; then
  echo "Could not parse total size from range header:" >&2
  cat "${header}" >&2
  exit 4
fi

echo "Current size: ${current_size}"
echo "Total size: ${total_size}"
echo "Chunk bytes: ${CHUNK_BYTES}"

part_path="${ZIP_PATH}.part"
while [[ "${current_size}" -lt "${total_size}" ]]; do
  end_byte=$((current_size + CHUNK_BYTES - 1))
  if [[ "${end_byte}" -ge "${total_size}" ]]; then
    end_byte=$((total_size - 1))
  fi
  expected_size=$((end_byte - current_size + 1))
  echo "Downloading bytes ${current_size}-${end_byte} at $(date)"
  rm -f "${part_path}"
  curl -L --fail --retry 20 --retry-delay 30 \
    -r "${current_size}-${end_byte}" \
    -o "${part_path}" \
    "${url}"
  actual_size="$(stat -c%s "${part_path}")"
  if [[ "${actual_size}" -ne "${expected_size}" ]]; then
    echo "Chunk size mismatch: expected ${expected_size}, got ${actual_size}" >&2
    exit 5
  fi
  cat "${part_path}" >> "${ZIP_PATH}"
  rm -f "${part_path}"
  current_size="$(stat -c%s "${ZIP_PATH}")"
  echo "Resumed zip size is now ${current_size} / ${total_size}"
done

echo "Login-node download finished at $(date)"
ls -lh "${ZIP_PATH}"
file "${ZIP_PATH}"
