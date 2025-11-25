#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
import h5py
from typing import Tuple, Dict

PAD_GENE = "NA"

def chr_to_num(chr_val: str) -> int:
    s = str(chr_val).strip()
    if s.startswith("chr"):
        s = s[3:]
    try:
        return int(s)
    except ValueError:
        return -1

def load_genes(genes_info_path: str) -> pd.DataFrame:
    df = pd.read_csv(genes_info_path, sep=r"\s+|\t", engine="python")
    cols = {c.lower(): c for c in df.columns}
    rename = {}
    for want, alts in {
        "gene_name": ["gene_name","gene","name","symbol"],
        "seqname":   ["seqname","chrom","chr","chromosome"],
        "start":     ["start","txStart","gene_start"],
        "end":       ["end","txEnd","gene_end"],
    }.items():
        for a in alts:
            if a in cols:
                rename[cols[a]] = want
                break
    df = df.rename(columns=rename)
    for required in ["gene_name","seqname","start","end"]:
        if required not in df.columns:
            raise ValueError(f"genesInfo missing required column '{required}'")
    df["chr_str"] = df["seqname"].astype(str)
    df["chr_num"] = df["chr_str"].apply(chr_to_num)
    df = df[df["chr_num"].between(1,22)].copy()
    df["start"] = df["start"].astype(int)
    df["end"]   = df["end"].astype(int)
    df = df.sort_values(["chr_num","start","end"]).reset_index(drop=True)
    return df[["gene_name","chr_str","chr_num","start","end"]]

def load_exons_bed(exons_bed_path: str) -> pd.DataFrame:
    bed = pd.read_csv(exons_bed_path, sep=r"\s+|\t", header=None, names=["chr","start","end"], engine="python")
    bed["chr_str"] = bed["chr"].astype(str)
    bed["chr_num"] = bed["chr_str"].apply(chr_to_num)
    bed = bed[bed["chr_num"].between(1,22)].copy()
    bed["start"] = bed["start"].astype(int)
    bed["end"]   = bed["end"].astype(int)
    bed = bed.sort_values(["chr_num","start","end"]).reset_index(drop=True)
    return bed[["chr","chr_str","chr_num","start","end"]]

def annotate_exons_with_genes(bed: pd.DataFrame, genes: pd.DataFrame) -> pd.DataFrame:
    bed = bed.copy()
    bed["gene_name"] = PAD_GENE
    bed["exon_id"]   = -1
    for chrom in sorted(bed["chr_num"].unique()):
        ex_idx = bed.index[bed["chr_num"] == chrom]
        if len(ex_idx) == 0: 
            continue
        exons_c = bed.loc[ex_idx]
        genes_c = genes[genes["chr_num"] == chrom]
        if len(genes_c) == 0:
            continue
        ex_start = exons_c["start"].to_numpy()
        ex_end   = exons_c["end"].to_numpy()
        for row in genes_c.itertuples(index=False):
            gs, ge = row.start, row.end
            mask = (ex_start >= gs) & (ex_end <= ge)
            if mask.any():
                bed.loc[ex_idx[mask], "gene_name"] = row.gene_name
        assigned = bed.loc[ex_idx].sort_values(["gene_name","start","end"])
        for gname, df_g in assigned.groupby("gene_name", sort=False):
            if gname == PAD_GENE:
                continue
            bed.loc[df_g.index, "exon_id"] = np.arange(1, len(df_g)+1, dtype=int)
    return bed

def write_annotated_bed(bed_annot: pd.DataFrame, out_path: str) -> None:
    out = bed_annot[["chr","start","end","gene_name","exon_id"]].copy()
    out.to_csv(out_path, sep="\t", index=False, header=True)

def build_exon_map_from_bed(bed_annot: pd.DataFrame) -> Dict[Tuple[int,int,int], Tuple[str,int]]:
    m: Dict[Tuple[int,int,int], Tuple[str,int]] = {}
    for r in bed_annot.itertuples(index=False):
        m[(int(r.chr_num), int(r.start), int(r.end))] = (r.gene_name, int(r.exon_id))
    return m

def first_sample_and_exons_order(h5in: h5py.File) -> Tuple[str, int, np.ndarray]:
    samples = list(h5in.keys())
    if not samples:
        raise RuntimeError("No sample groups found in input HDF5.")
    s0 = samples[0]
    if "meta" not in h5in[s0]:
        raise RuntimeError(f"Sample '{s0}' missing 'meta' dataset.")
    meta0 = np.array(h5in[s0]["meta"])
    if meta0.ndim != 2 or meta0.shape[1] < 3:
        raise RuntimeError(f"'meta' has unexpected shape {meta0.shape} for sample '{s0}'.")
    return s0, meta0.shape[0], meta0

def gene_arrays_for_meta(meta: np.ndarray, exon_map: Dict[Tuple[int,int,int], Tuple[str,int]]) -> Tuple[np.ndarray, np.ndarray]:
    starts = meta[:,0].astype(int)
    ends   = meta[:,1].astype(int)
    chrs   = meta[:,2].astype(int)
    n = meta.shape[0]
    gene_names = np.empty(n, dtype=object)
    exon_ids   = np.full(n, -1, dtype=np.int32)
    for i in range(n):
        val = exon_map.get((int(chrs[i]), int(starts[i]), int(ends[i])))
        if val is None:
            gene_names[i] = PAD_GENE
            exon_ids[i]   = -1
        else:
            gene_names[i], exon_ids[i] = val[0], int(val[1])
    return gene_names, exon_ids

def write_gene_meta_to_sample(h5file: h5py.File, sample: str, gene_names: np.ndarray, exon_ids: np.ndarray):
    grp = h5file[sample]
    if "gene_name" in grp:
        del grp["gene_name"]
    if "exon_id" in grp:
        del grp["exon_id"]
    str_dt = h5py.string_dtype(encoding="utf-8")
    grp.create_dataset("gene_name", data=gene_names.astype(str_dt), compression="gzip")
    grp.create_dataset("exon_id",   data=exon_ids.astype(np.int32),  compression="gzip")

def add_top_level_exon_table(h5file: h5py.File, bed_annot: pd.DataFrame):
    # overwrite if it exists
    if "exon_annotation" in h5file:
        del h5file["exon_annotation"]
    g = h5file.create_group("exon_annotation")

    # variable-length UTF-8 for strings
    dt_str = h5py.string_dtype(encoding="utf-8")

    # helper: write a string Series safely
    def write_str(name, series):
        arr_obj = series.astype(str).to_numpy(dtype=object)  # object array of Python str
        g.create_dataset(name, data=arr_obj, dtype=dt_str, compression="gzip")

    # numeric fields
    g.create_dataset("chr_num", data=bed_annot["chr_num"].to_numpy(dtype=np.int32), compression="gzip")
    g.create_dataset("start",   data=bed_annot["start"].to_numpy(dtype=np.int32),   compression="gzip")
    g.create_dataset("end",     data=bed_annot["end"].to_numpy(dtype=np.int32),     compression="gzip")
    g.create_dataset("exon_id", data=bed_annot["exon_id"].to_numpy(dtype=np.int32), compression="gzip")

    # string fields
    write_str("chr",       bed_annot["chr"])
    write_str("gene_name", bed_annot["gene_name"])


def main():
    ap = argparse.ArgumentParser(description="In-place: add gene_name & exon_id to each sample in HDF5; also write annotated BED TSV.")
    ap.add_argument("--genes-info", default="/")
    ap.add_argument("--exons-bed", default="/")
    ap.add_argument("--in-h5", default="/")
    ap.add_argument("--out-bed", default="/")
    ap.add_argument("--write-top-level", action="store_true", help="Also store a canonical /exon_annotation table in the same HDF5.")
    args = ap.parse_args()

    print("Loading genes...")
    genes = load_genes(args.genes_info)
    print(f"  Loaded {len(genes):,} genes (autosomes only).")

    print("Loading exons (BED)...")
    bed = load_exons_bed(args.exons_bed)
    print(f"  Loaded {len(bed):,} exons.")

    print("Annotating exons with gene_name and exon_id...")
    bed_annot = annotate_exons_with_genes(bed, genes)
    annotated_count = (bed_annot["gene_name"] != PAD_GENE).sum()
    print(f"  Annotated {annotated_count:,} / {len(bed_annot):,} exons.")

    print(f"Writing annotated BED/TSV to: {args.out_bed}")
    write_annotated_bed(bed_annot, args.out_bed)

    print("Building exon map (chr_num,start,end -> gene_name,exon_id)...")
    exon_map = build_exon_map_from_bed(bed_annot)

    print(f"Opening input HDF5 for in-place write: {args.in_h5}")
    with h5py.File(args.in_h5, "r+") as h5in:
        s0, n_exons, meta0 = first_sample_and_exons_order(h5in)
        print(f"  First sample: {s0}, exons: {n_exons:,}")

        gn0, ex0 = gene_arrays_for_meta(meta0, exon_map)
        na_missing = int((ex0 == -1).sum())
        if na_missing > 0:
            print(f"WARNING: {na_missing:,} exons in first sample did not match any gene span; "
                  f"they will get gene_name='{PAD_GENE}', exon_id=-1.")

        samples = list(h5in.keys())
        for si, sample in enumerate(samples, 1):
            if "meta" not in h5in[sample]:
                print(f"  Skipping sample {sample}: no 'meta'")
                continue
            m = np.array(h5in[sample]["meta"])
            if m.shape[0] != n_exons:
                gn, exid = gene_arrays_for_meta(m, exon_map)
            else:
                gn, exid = gn0, ex0
            write_gene_meta_to_sample(h5in, sample, gn, exid)
            if si % 10 == 0 or si == len(samples):
                print(f"  Processed {si}/{len(samples)} samples...")

        if args.write_top_level:
            print("Writing top-level /exon_annotation table into the same HDF5...")
            add_top_level_exon_table(h5in, bed_annot)

    print("Done (in-place).")

if __name__ == "__main__":
    main()
