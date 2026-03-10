#!/bin/bash

set -e

mkdir -p data
mkdir -p results

SIZES=(
  67108864
  134217728
  268435456
  536870912
)

LOCAL_SIZES=(
  64
  128
  256
)

for SIZE in "${SIZES[@]}"; do
  FILE="data/test_${SIZE}.bin"

  if [ ! -f "$FILE" ]; then
    python3 scripts/generate_data.py "$FILE" "$SIZE" 0.1
  fi

  for LS in "${LOCAL_SIZES[@]}"; do
    echo "Running for file=$FILE local_size=$LS"
    ./zero_count "$FILE" "$LS"
  done
done