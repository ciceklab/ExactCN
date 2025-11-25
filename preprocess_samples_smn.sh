#!/bin/bash

# --- 1. Define Variables ---
INPUT_DIR="./finetune_example_data/bam_files"
READ_DEPTH_DIR="./read_depths"
NPY_OUTPUT_DIR="./processed_samples_npy"
FINAL_H5_FILE="./smn_data/smn_samples.h5"

SMN_OUTPUT_H5="./smn_data/smn_samples.h5"

TARGET_BED="hglft_genome_64dc_dcbaa0_unique.bed"
ANNOTATED_BED="./smn_data/annotated_targets.tsv"
GENES_INFO_FILE="genesInfo.txt" 

# --- 2. Setup Directories ---
mkdir -p "$READ_DEPTH_DIR"
mkdir -p "$NPY_OUTPUT_DIR"
mkdir -p "$(dirname "$FINAL_H5_FILE")"

# --- 3. Run Standard Pipeline ---
echo "Starting Step 1: BAM Indexing and Depth Calculation..."
for filename in "$INPUT_DIR"/*.bam; do
    [ -e "$filename" ] || continue
    base_name=$(basename "$filename" .bam)
    echo "Processing: $base_name"

    if [ ! -f "${filename}.bai" ]; then
        echo "  - Indexing..."
        samtools index "$filename"
    fi

    output_depth="$READ_DEPTH_DIR/${base_name}.txt.gz"
    if [ ! -f "$output_depth" ]; then
        echo "  - Calculating depth..."
        sambamba depth base -L "$TARGET_BED" "$filename" | gzip > "$output_depth"
    else
        echo "  - Depth file already exists, skipping."
    fi
done

echo "Starting Step 2: Generating NPY files..."
python3 ./scripts/preprocess_data_exactcn.py \
    --rd_dir "$READ_DEPTH_DIR" \
    --targets "$TARGET_BED" \
    --out "$NPY_OUTPUT_DIR" \
    --threads 8

echo "Starting Step 3: Compressing to HDF5..."
python3 ./scripts/compress_to_h5.py \
    "$NPY_OUTPUT_DIR" \
    "$FINAL_H5_FILE" \
    --compression gzip \
    --level 4

echo "Starting Step 4: Annotating HDF5 with Gene Names..."
python3 ./scripts/annotate_exons.py \
    --in-h5 "$FINAL_H5_FILE" \
    --exons-bed "$TARGET_BED" \
    --genes-info "$GENES_INFO_FILE" \
    --out-bed "$ANNOTATED_BED" \
    --write-top-level



echo "Starting Step 5: Filtering SMN1/SMN2 and Merging Signals..."
python3 ./scripts/merge_smn.py \
    --input "$FINAL_H5_FILE" \
    --output "$SMN_OUTPUT_H5"

echo "------------------------------------------------"
echo "Pipeline Finished!"
echo "Standard Output: $FINAL_H5_FILE (All genes)"
echo "SMN Specific Output: $SMN_OUTPUT_H5 (Only merged to SMN2)"