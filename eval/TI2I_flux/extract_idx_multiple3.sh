#!/bin/bash

SRC_DIR="/Volumes/home/T2I/Concept_Erasure/SexualErasure_CCS/eval/TI2I_flux/results"
DST_DIR="/Volumes/home/T2I/Concept_Erasure/SexualErasure_CCS/eval/TI2I_flux/tmp"

mkdir -p "$DST_DIR"

count=0
for f in "$SRC_DIR"/idx*.png; do
    filename=$(basename "$f")
    # Extract the idx number from filename like idx0003_sid1.png
    idx_num=$(echo "$filename" | sed -n 's/idx\([0-9]*\)_.*/\1/p')
    # Remove leading zeros for arithmetic
    idx=$((10#$idx_num))
    if (( idx % 3 == 0 )); then
        cp "$f" "$DST_DIR/"
        count=$((count + 1))
    fi
done

echo "Copied $count files where idx is a multiple of 3 to $DST_DIR"
