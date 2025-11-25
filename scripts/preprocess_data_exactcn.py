#!/usr/bin/env python3
"""
Pre-processing for ECOLÉ-Genotyper regression
UPDATED: Supports Inference Mode (No Ground Truth Labels required)

Output
------
signals      float32  (N_exons, 4, 1000)   A,T,C,G coverage
meta         int32    (N_exons, 3)         chr, start, end  (1-based)
copy_number  int8     (N_exons,)           summed CN (Training: Real values | Inference: -1)
gene_name    bytes    (N_exons,)           Gene Name (Default: NA)
"""

import os, argparse, logging, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd

#CLI
argp = argparse.ArgumentParser()
argp.add_argument("--rd_dir",   required=True, help="*.txt.gz read-depth files")
argp.add_argument("--targets",  required=True, help="BED/TSV chr start end")
# CHANGED: labels is now OPTIONAL
argp.add_argument("--labels",   default=None,  help="Groundtruth_*.csv folder (Optional for inference)")
argp.add_argument("--out",      required=True, help="Output folder for .npy")
argp.add_argument("--threads",  type=int, default=4)
args = argp.parse_args()
os.makedirs(args.out, exist_ok=True)

#helper
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")

NT_COLS  = ["A", "T", "C", "G"]
PAD_VAL  = -1.0
SLICE_SZ = 1000

# Load Targets (Inject dummy NA for gene_name if missing)
targets = (
    pd.read_csv(args.targets, sep="\t", header=None,
                names=["chr", "start", "end"])
      .astype({"start":"int32", "end":"int32"})
)
targets = targets[~targets.chr.isin(["chrX", "chrY"])].reset_index(drop=True)
targets["gene_name"] = "NA" 

def label_token_to_cn(tok:str) -> int:
    tok = tok.strip("<>").upper()
    return int(tok[2:]) if tok.startswith("CN") else 1 # diploid default

def build_signal(df_chr:pd.DataFrame, start:int, end:int) -> np.ndarray:
    """Return a (4,1000) float32 matrix; crop if exon > 1000 bp."""
    exon_len = end - start
    sig = np.full((4, SLICE_SZ), PAD_VAL, dtype=np.float32)

    if exon_len == 0:
        sig[:] = 0.0
        return sig

    if exon_len > SLICE_SZ:
        trim = exon_len - SLICE_SZ
        start += trim
        exon_len = SLICE_SZ

    tmp = np.zeros((4, exon_len), dtype=np.float32)
    rows = df_chr[(df_chr.POS >= start) & (df_chr.POS < end)]
    if not rows.empty:
        rel = (rows.POS.values - start).astype("int32")
        for i, nt in enumerate(NT_COLS):
            tmp[i, rel] = rows[nt].values

    sig[:, SLICE_SZ - exon_len :] = tmp 
    return sig

def process_sample(rd_file: str, idx: int, total: int):
    sample  = rd_file.split(".", 1)[0]
    out_npy = os.path.join(args.out, f"{sample}.npy")

    if os.path.exists(out_npy):
        try:
            existing = np.load(out_npy, allow_pickle=True).item()
            assert isinstance(existing, dict)
            for key in ("signals","meta","copy_number", "gene_name"): 
                assert key in existing
        except Exception:
            logging.warning(f"[{idx}/{total}] {sample} – existing .npy corrupted, re-processing")
            try: os.remove(out_npy)
            except OSError: pass
        else:
            logging.info(f"[{idx}/{total}] {sample} – already done, skipping")
            return

    try:
        logging.info(f"[{idx}/{total}] {sample} – reading RD")
        rd_df = pd.read_csv(
            os.path.join(args.rd_dir, rd_file),
            sep="\t", compression="gzip",
            usecols=["REF","POS"] + NT_COLS
        )
        rd_df.POS = rd_df.POS.astype("int32")
        rd_by_chr = {c: df for c, df in rd_df.groupby("REF")}

        # --- LOGIC BRANCH: Do we have labels? ---
        label_file = os.path.join(args.labels, f"Groundtruth_{sample}.csv") if args.labels else None
        
        has_groundtruth = False
        if label_file and os.path.exists(label_file):
            has_groundtruth = True
        
        if has_groundtruth:
            # === TRAINING MODE ===
            lab_df = pd.read_csv(label_file, sep="\t")
            lab_df["cn"] = (
                lab_df.label_parent_0.map(label_token_to_cn) +
                lab_df.label_parent_1.map(label_token_to_cn)
            )
            # Inner merge: Only keep targets that have labels
            merged = (
                targets
                  .merge(
                    lab_df[["target_chr","target_start","target_end","cn"]],
                    left_on  = ["chr","start","end"],
                    right_on = ["target_chr","target_start","target_end"],
                    how="inner"
                  )
                  .loc[:, ["chr","start","end","cn","gene_name"]]
            )
        else:
            # === INFERENCE MODE ===
            # No labels? Use all targets and set CN = -1 (Dummy)
            merged = targets.copy()
            merged["cn"] = -1 # Placeholder value for unknown CN
            # Ensure correct column order
            merged = merged[["chr", "start", "end", "cn", "gene_name"]]

        # Group by chromosome for processing
        intervals_by_chr = {
            chrom: list(zip(g.start.values, g.end.values, g.cn.values, g.gene_name.values))
            for chrom, g in merged.groupby("chr")
        }

        sigs, metas, cns, genes = [], [], [], []
        for chrom, recs in intervals_by_chr.items():
            df_chr = rd_by_chr.get(chrom)
            if df_chr is None:
                continue
            for start, end, cn, gn in recs:
                sigs .append(build_signal(df_chr, start, end))
                metas.append([start, end, int(chrom.lstrip("chr"))])
                cns  .append(cn)
                genes.append(gn)

        if not sigs:
            logging.warning(f"[{idx}/{total}] {sample} – no exons found")
            return

        np.save(out_npy, {
            "signals"     : np.stack(sigs).astype(np.float32),
            "meta"        : np.asarray(metas, dtype=np.int32),
            "copy_number" : np.asarray(cns,   dtype=np.int8),
            "gene_name"   : np.asarray(genes, dtype='S')
        }, allow_pickle=True)

        mode_str = "Training" if has_groundtruth else "Inference"
        logging.info(f"[{idx}/{total}] {sample} – written ({len(sigs)} exons) [{mode_str} Mode]")

    except Exception:
        logging.error(f"[{idx}/{total}] {sample} – FAILED\n{traceback.format_exc()}")

def main():
    rd_files = sorted(f for f in os.listdir(args.rd_dir) if f.endswith(".gz"))
    logging.info(f"{len(rd_files):,} RD files detected")

    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        fut2file = {pool.submit(process_sample, f, i+1, len(rd_files)): f
                    for i, f in enumerate(rd_files)}
        for fut in as_completed(fut2file):
            fut.result()

if __name__ == "__main__":
    main()