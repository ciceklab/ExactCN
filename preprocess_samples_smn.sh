#!/bin/bash

INPUT_DIR="./example_data"
READ_DEPTH_DIR="./read_depths"
NPY_OUTPUT_DIR="./processed_samples_npy"
LABELS_DIR="./ground_truth_labels"
TARGET_BED="hglft_genome_64dc_dcbaa0.bed"

FINAL_H5_FILE="./data/all_samples_updated.h5"
SMN_H5_FILE="./data/smn_samples.h5"

SMN_START=69000000
SMN_END=71000000

mkdir -p "$READ_DEPTH_DIR"
mkdir -p "$NPY_OUTPUT_DIR"
mkdir -p "$(dirname "$FINAL_H5_FILE")"

echo "Step 1: BAM Processing"
for filename in "$INPUT_DIR"/*.bam; do
    [ -e "$filename" ] || continue
    base_name=$(basename "$filename" .bam)
    
    if [ ! -f "${filename}.bai" ]; then
        samtools index "$filename"
    fi

    output_depth="$READ_DEPTH_DIR/${base_name}.txt.gz"
    if [ ! -f "$output_depth" ]; then
        sambamba depth base -L "$TARGET_BED" "$filename" | gzip > "$output_depth"
    fi
done

echo "Step 2: Generating NPY"
python3 ./scripts/preprocess_data_exactcn.py \
    --rd_dir "$READ_DEPTH_DIR" \
    --targets "$TARGET_BED" \
    --labels "$LABELS_DIR" \
    --out "$NPY_OUTPUT_DIR" \
    --threads 8

echo "Step 3: Creating Master H5"
python3 ./scripts/compress_to_h5.py \
    "$NPY_OUTPUT_DIR" \
    "$FINAL_H5_FILE" \
    --compression gzip \
    --level 4

echo "Step 4: Filtering for SMN Region"
python3 ./scripts/filter_h5_smn.py \
    --input "$FINAL_H5_FILE" \
    --output "$SMN_H5_FILE" \
    --start "$SMN_START" \
    --end "$SMN_END"

echo "Pipeline Finished."