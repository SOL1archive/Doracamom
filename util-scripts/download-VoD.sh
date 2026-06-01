#!/bin/bash
set -euo pipefail

USERNAME="${VOD_USERNAME:-subin}"
OUTPUT_DIR="${VOD_OUTPUT_DIR:-.}"
BASE_URL="https://viewofdelft-dataset.tudelft.nl/dataset/download/basic"

if [[ -z "${VOD_PASSWORD:-}" ]]; then
    read -rsp "Enter your password: " PASSWORD
    echo
else
    PASSWORD="$VOD_PASSWORD"
fi
WGET_CONFIG="$(mktemp)"
trap 'rm -f "$WGET_CONFIG"' EXIT
chmod 600 "$WGET_CONFIG"
cat > "$WGET_CONFIG" <<EOF
user = $USERNAME
password = $PASSWORD
EOF

mkdir -p "$OUTPUT_DIR"

FILES=(
    "view_of_delft_detection_PUBLIC.zip"
    "view_of_delft_prediction_PUBLIC.zip"
    "label_2_with_track_ids.zip"
)

for FILE in "${FILES[@]}"; do
    echo "Downloading $FILE..."
    wget -c --tries=0 --waitretry=10 --timeout=120 \
        --config="$WGET_CONFIG" \
        -P "$OUTPUT_DIR" "$BASE_URL/$FILE"
done
