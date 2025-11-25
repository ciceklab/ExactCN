#!/usr/bin/env python3
import os
import argparse
import time
import csv
import math
from multiprocessing import Process, Queue
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from performer_pytorch import Performer

# Constants
D_MODEL = 512
DEPTH = 5
HEADS = 8
EXON_SIZE = 1000
CHANNELS = 4
PAD_VAL = -1.0
MAX_EXON_ID = 128
AUTOSOMES = set(range(1, 23))
DEL_CUTOFF = 1.60
DUP_CUTOFF = 2.20
CLASS_LABELS = ["DEL", "NO-CALL", "DUP"]

# Global normalization stats
GLOBAL_MEAN = None
GLOBAL_STD = None

# Resume helpers
def expected_rows_for_sample(h5_path: str, sample_name: str) -> int:
    try:
        with h5py.File(h5_path, 'r') as f:
            if sample_name not in f:
                return 0
            grp = f[sample_name]
            if not isinstance(grp, h5py.Group):
                return 0
            if 'meta' not in grp:
                return 0
            meta = grp['meta'][:]
            if meta.ndim != 2 or meta.shape[1] < 3:
                return 0
            return int(np.isin(meta[:, 2], list(AUTOSOMES)).sum())
    except Exception:
        return 0

def completed_rows_in_csv(path: str) -> int:
    try:
        with open(path, 'r') as f:
            n = sum(1 for _ in f)
        return max(0, n - 1)
    except Exception:
        return -1

def out_paths_for(h5_path: str, sample_name: str, outdir: str):
    base = os.path.splitext(os.path.basename(h5_path))[0]
    legacy = os.path.join(outdir, f"{sample_name}.csv")
    disamb = os.path.join(outdir, f"{base}__{sample_name}.csv")
    return legacy, disamb

# Normalization
def load_normalization(norm_file: str, device, verbose=True):
    """Load per-channel normalization stats (same as training)."""
    global GLOBAL_MEAN, GLOBAL_STD
    try:
        rec = np.genfromtxt(norm_file, delimiter=',', names=True, dtype=float, max_rows=1)
        fields = [
            'signal_mean_A','signal_std_A',
            'signal_mean_T','signal_std_T',
            'signal_mean_C','signal_std_C',
            'signal_mean_G','signal_std_G'
        ]
        if all(name in rec.dtype.names for name in fields):
            means = np.array([rec['signal_mean_A'], rec['signal_mean_T'],
                              rec['signal_mean_C'], rec['signal_mean_G']], dtype=np.float32)
            stds  = np.array([rec['signal_std_A'],  rec['signal_std_T'],
                              rec['signal_std_C'],  rec['signal_std_G']], dtype=np.float32)
        else:
            raise ValueError("Header mismatch")
    except Exception:
        arr = np.loadtxt(norm_file, delimiter=',').astype(np.float32).ravel()
        if arr.size < 8:
            raise RuntimeError(f"Normalization file {norm_file} does not contain 8 numbers.")
        means = np.array([arr[0], arr[2], arr[4], arr[6]], dtype=np.float32)
        stds  = np.array([arr[1], arr[3], arr[5], arr[7]], dtype=np.float32)

    stds = np.clip(stds, 1e-6, None)
    GLOBAL_MEAN = torch.tensor(means, device=device, dtype=torch.float32)
    GLOBAL_STD  = torch.tensor(stds,  device=device, dtype=torch.float32)

    if verbose:
        print("Loaded per-channel normalization (A,T,C,G):")
        for c, m, s in zip(['A','T','C','G'], means, stds):
            print(f"  {c}: mean={m:.6f}, std={s:.6f}")

def normalize_signals(sig: torch.Tensor) -> torch.Tensor:
    assert GLOBAL_MEAN is not None and GLOBAL_STD is not None
    mask = (sig != PAD_VAL)
    sig_n = sig.clone()
    mu = GLOBAL_MEAN.view(1, CHANNELS, 1)
    sd = GLOBAL_STD.view(1, CHANNELS, 1)
    sig_n[mask] = ((sig_n - mu) / sd)[mask]
    sig_n[~mask] = PAD_VAL
    return sig_n

# Vocab I/O
def load_gene_vocab_from_file(vocab_file):
    gene_vocab = []
    with open(vocab_file, 'r') as f:
        for line in f:
            gene_vocab.append(line.strip())
    gene2id = {gene: i for i, gene in enumerate(gene_vocab)}
    return gene_vocab, gene2id

def build_gene_vocab_from_h5(h5_path):
    gene_set = set()
    with h5py.File(h5_path, 'r') as f:
        for sample_name in f.keys():
            obj = f[sample_name]
            if isinstance(obj, h5py.Group) and 'gene_name' in obj:
                genes = obj['gene_name'][:]
                if isinstance(genes, np.ndarray) and genes.dtype.kind in ['U', 'S']:
                    genes = [g.decode() if isinstance(g, bytes) else str(g) for g in genes]
                gene_set.update(genes)
    gene_vocab = ['NA'] + sorted(gene_set - {'NA'})
    gene2id = {gene: i for i, gene in enumerate(gene_vocab)}
    return gene_vocab, gene2id

# Model (regression-only; must mirror training)
class LocalSinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len=EXON_SIZE):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)
    def forward(self, B): return self.pe.unsqueeze(0).expand(B, -1, -1)

class CNNTokenizer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.conv1 = nn.Conv1d(CHANNELS, 64,  kernel_size=9, padding=4)
        self.conv2 = nn.Conv1d(64,      128, kernel_size=9, padding=4)
        self.conv3 = nn.Conv1d(128,     d_model, kernel_size=1)
        self.act   = nn.GELU()
        self.norm_ch  = nn.GroupNorm(num_groups=32, num_channels=d_model) # (B, D, N)
        self.norm_tok = nn.LayerNorm(d_model) # (B, N, D)

    def forward(self, x, mask):
        # x: (B,4,1000); mask: (B,1000) for "valid tokens"
        x = x.clone()
        x[~torch.isfinite(x)] = 0.0
        x[x == PAD_VAL] = 0.0
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x) # (B, D, N)
        x = self.norm_ch(x)
        x = rearrange(x, 'b d n -> b n d') # (B, N, D)
        x = self.norm_tok(x)
        x = x * mask.unsqueeze(-1).float()
        return x

class CNVRegressor(nn.Module):
    def __init__(self, n_genes):
        super().__init__()
        self.tok = CNNTokenizer(D_MODEL)
        self.emb_chr  = nn.Embedding(23, D_MODEL) # 0..22
        self.emb_gene = nn.Embedding(n_genes + 1, D_MODEL) # 0 = NA/UNK
        self.emb_exon = nn.Embedding(MAX_EXON_ID + 1, D_MODEL)
        self.pos = LocalSinusoidalPositionalEmbedding(D_MODEL)
        self.cls = nn.Parameter(torch.randn(1, 1, D_MODEL))
        self.backbone = Performer(dim=D_MODEL, depth=DEPTH, heads=HEADS, dim_head=64,
                                  ff_dropout=0.1, attn_dropout=0.1)
        self.reg_head = nn.Sequential(
            nn.LayerNorm(D_MODEL),
            nn.Linear(D_MODEL, D_MODEL),
            nn.GELU(),
            nn.Linear(D_MODEL, 1)
        )

    def forward(self, sig_n, meta, gene_id, exon_id):
        B = sig_n.size(0)
        valid_mask = torch.any(sig_n != PAD_VAL, dim=1)
        tok = self.tok(sig_n, valid_mask)
        e_chr = self.emb_chr(meta[:, 2].long().clamp(min=0, max=22)).unsqueeze(1)
        e_gene = self.emb_gene(gene_id.clamp(min=0)).unsqueeze(1)
        e_exon = self.emb_exon(exon_id.long().clamp(min=0, max=MAX_EXON_ID)).unsqueeze(1)
        e_ctx = e_chr + e_gene + e_exon
        x = tok + self.pos(B) + e_ctx
        cls_tok = self.cls.expand(B, 1, -1) + e_ctx
        seq = torch.cat([cls_tok, x], dim=1)
        mask = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=seq.device), valid_mask], dim=1)
        enc = self.backbone(seq, mask=mask)
        cls_enc = enc[:, 0]
        pred_reg = self.reg_head(cls_enc).squeeze(1)
        return pred_reg

# Weights
def load_weights_flexible(model: nn.Module, ckpt_path: str, device, verbose=True):
    sd = torch.load(ckpt_path, map_location=device)

    def _try_load(state_dict, note):
        incompat = model.load_state_dict(state_dict, strict=False)
        if verbose:
            print(f"Loaded weights ({note}).")
            if getattr(incompat, "missing_keys", None):
                print("  Missing keys:", incompat.missing_keys)
            if getattr(incompat, "unexpected_keys", None):
                print("  Unexpected keys:", incompat.unexpected_keys)

    try:
        _try_load(sd, "as-is")
        return
    except Exception as e:
        if verbose:
            print(f"Direct load failed: {e}  -> trying key prefix adjustment")

    # Try removing "module." prefix
    new_sd = {}
    for k, v in sd.items():
        new_sd[k[7:]] = v if k.startswith("module.") else v
    try:
        _try_load(new_sd, "stripped 'module.'")
        return
    except Exception as e2:
        if verbose:
            print(f"Strip 'module.' also failed: {e2} -> trying to add 'module.'")

    # Try adding "module." prefix
    new_sd = {}
    for k, v in sd.items():
        new_sd[f"module.{k}"] = v if not k.startswith("module.") else v
    _try_load(new_sd, "added 'module.'")

# Helpers
def cn_array_to_index(cn_values: np.ndarray) -> np.ndarray:
    tri = np.full(cn_values.shape, 1, dtype=np.int64)
    tri[cn_values <= DEL_CUTOFF] = 0
    tri[cn_values >= DUP_CUTOFF] = 2
    return tri

# Inference
@torch.no_grad()
def predict_one_sample(h5_path: str, sample_name: str, model: nn.Module, device, batch_size: int, gene2id: dict):
    with h5py.File(h5_path, 'r') as h5f:
        grp = h5f[sample_name]
        N = grp['meta'].shape[0]

        meta_all = grp['meta'][:]  # (N, 3) int32: [start, end, chr]
        mask_auto = np.isin(meta_all[:, 2], list(AUTOSOMES))

        # gene names / exon ids
        if 'gene_name' in grp:
            gene_names_raw = grp['gene_name'][:]
            gene_names = [
                g.decode() if isinstance(g, (bytes, np.bytes_)) else str(g)
                for g in gene_names_raw
            ]
        else:
            gene_names = ['NA'] * N

        exon_ids = grp['exon_id'][:] if 'exon_id' in grp else np.zeros(N, dtype=np.int32)

        results = []
        processed_exons = 0
        print(f"  [{sample_name}] Processing {N:,} exons in batches of {batch_size}")

        for batch_idx, start_idx in enumerate(range(0, N, batch_size)):
            end_idx = min(start_idx + batch_size, N)

            submask = mask_auto[start_idx:end_idx]
            if not np.any(submask):
                processed_exons += end_idx - start_idx
                continue

            rel = np.nonzero(submask)[0]
            idxs = start_idx + rel

            sig_np  = grp['signals'][idxs].astype(np.float32)
            meta_np = meta_all[idxs].astype(np.int32)

            batch_gene_names = [gene_names[i] for i in idxs]
            batch_gene_ids = np.array([gene2id.get(g, 0) for g in batch_gene_names], dtype=np.int64)
            batch_exon_ids = exon_ids[idxs]

            sig = torch.from_numpy(sig_np).to(device, non_blocking=True)
            meta = torch.from_numpy(meta_np).to(device, non_blocking=True)
            gene_ids = torch.from_numpy(batch_gene_ids).to(device, non_blocking=True)
            exon_ids_t = torch.from_numpy(batch_exon_ids).to(device, non_blocking=True)

            sig_n = normalize_signals(sig)

            pred_reg = model(sig_n, meta, gene_ids, exon_ids_t)

            pred_reg_np = pred_reg.detach().cpu().numpy()   # shape (B,)
            pred_reg_idx = cn_array_to_index(pred_reg_np)

            # free tensors
            del sig, meta, gene_ids, exon_ids_t, sig_n, pred_reg

            for i in range(len(pred_reg_idx)):
                c  = int(meta_np[i, 2])
                st = int(meta_np[i, 0])
                en = int(meta_np[i, 1])
                gn = batch_gene_names[i]
                reg_class = CLASS_LABELS[int(pred_reg_idx[i])]
                reg_value_fmt = f"{float(pred_reg_np[i]):.2f}"  # x.xx
                results.append((c, st, en, gn, reg_class, reg_value_fmt))

            del pred_reg_np, pred_reg_idx
            processed_exons += len(idxs)

            if batch_idx % 50 == 0 and batch_idx > 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

            if batch_idx % 20 == 0:
                progress_pct = (processed_exons / N) * 100
                print(f"    [{sample_name}] Batch {batch_idx+1}: {processed_exons:,}/{N:,} exons ({progress_pct:.1f}%)")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  [{sample_name}] Completed {len(results):,} autosomal exon predictions")
        return sample_name, results

# Worker (resume-aware, atomic writes)
def worker_process(h5_path, sample_name, model_path, norm_file, gene2id,
                   batch_size, gpu_id, out_path, expected_rows, result_queue):
    """Worker process for parallel sample processing (skips complete, atomic write)."""
    try:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Skip if already complete (double-check)
        if os.path.exists(out_path) and completed_rows_in_csv(out_path) >= expected_rows:
            result_queue.put((sample_name, expected_rows, out_path, None))
            return

        # If nothing to do (no autosomal exons), just mark as OK with 0 rows
        if expected_rows <= 0:
            result_queue.put((sample_name, 0, out_path, None))
            return

        load_normalization(norm_file, device, verbose=False)
        n_genes = len(gene2id) - 1
        model = CNVRegressor(n_genes).to(device)
        load_weights_flexible(model, model_path, device, verbose=False)
        model.eval()

        _, predictions = predict_one_sample(
            h5_path, sample_name, model, device, batch_size, gene2id
        )

        tmp_path = out_path + ".tmp"
        with open(tmp_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Regression-only output
            writer.writerow(['chr', 'start', 'end', 'gene_name', 'reg_class', 'reg_value'])
            writer.writerows(predictions)
        os.replace(tmp_path, out_path)

        result_queue.put((sample_name, len(predictions), out_path, None))

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        result_queue.put((sample_name, 0, None, str(e)))

# CLI
def parse_args():
    ap = argparse.ArgumentParser(description="CNV regression-only model test (parallel, resume-safe)")
    ap.add_argument("-i", "--input", required=True, nargs='+',
                    help="Test HDF5 file(s) - can specify multiple files")
    ap.add_argument("-n", "--norm_file", required=True, help="Normalization CSV")
    ap.add_argument("-m", "--model", required=True, help="Path to model checkpoint (.pt)")
    ap.add_argument("-v", "--vocab", help="Optional: direct path to gene_vocab.txt (preferable)")
    ap.add_argument("-o", "--output", required=True, help="Output directory")
    ap.add_argument("-bs", "--batch_size", default=64, type=int, help="Batch size (default: 64)")
    ap.add_argument("-g", "--gpu", default="0", help="GPU IDs (e.g., '0' or '0,1,2')")
    ap.add_argument("-j", "--jobs", default=2, type=int, help="Number of parallel workers")
    return ap.parse_args()

# Main
def main():
    args = parse_args()
    t_start = time.time()

    print("=== CNV Regression-Only Test (Parallel, Resume-Safe) ===")
    print(f"Model: {args.model}")
    print(f"Test data: {len(args.input)} file(s)")
    print(f"Output: {args.output}")
    print(f"Batch size: {args.batch_size}")

    gpu_list = [int(g.strip()) for g in args.gpu.split(',')]
    print(f"GPUs: {gpu_list} ({len(gpu_list)} GPU(s))")
    print(f"Parallel workers: {args.jobs} (distributed across {len(gpu_list)} GPU(s))")
    print(f"  ~{args.jobs / max(1,len(gpu_list)):.1f} workers per GPU (approx.)")

    # Vocab resolution
    vocab_file = None
    if args.vocab and os.path.exists(args.vocab):
        vocab_file = args.vocab
        print(f"\nUsing vocabulary from: {vocab_file} (user-specified)")
    else:
        model_dir = os.path.dirname(args.model)
        candidate = os.path.join(model_dir, "gene_vocab.txt")
        if os.path.exists(candidate):
            vocab_file = candidate
            print(f"\nUsing vocabulary from: {vocab_file} (model directory)")
        else:
            common_paths = [
                "/data6/erfan/ECOLE_CN/models/bertstyle_v2/gene_vocab.txt",
                os.path.join(os.path.dirname(model_dir), "bertstyle_v2", "gene_vocab.txt"),
            ]
            for candidate in common_paths:
                if os.path.exists(candidate):
                    vocab_file = candidate
                    print(f"\nUsing vocabulary from: {vocab_file} (found in training directory)")
                    break

    if vocab_file:
        gene_vocab, gene2id = load_gene_vocab_from_file(vocab_file)
        print("Using saved vocabulary for consistency")
    else:
        print(f"\nWARNING: Gene vocabulary not found! Building from test data: {args.input[0]}")
        print("   This may cause issues if genes differ from training! Use -v to specify gene_vocab.txt")
        gene_vocab, gene2id = build_gene_vocab_from_h5(args.input[0])
    print(f"Gene vocabulary size: {len(gene_vocab)}")

    os.makedirs(args.output, exist_ok=True)

    # Gather *valid* samples only (skip non-sample groups like 'exon_annotation')
    file_sample_pairs = []
    for h5_path in args.input:
        with h5py.File(h5_path, 'r') as h5f:
            for sample_name in h5f.keys():
                obj = h5f[sample_name]
                if not isinstance(obj, h5py.Group):
                    continue
                if 'meta' not in obj or 'signals' not in obj:
                    print(f"  IGNORE [{sample_name}] (no 'meta' and/or 'signals')")
                    continue
                meta = obj['meta']
                if meta.ndim != 2 or meta.shape[1] < 3:
                    print(f"  IGNORE [{sample_name}] ('meta' bad shape: {meta.shape})")
                    continue
                file_sample_pairs.append((h5_path, sample_name))

    print(f"\nFound {len(file_sample_pairs)} valid samples total across {len(args.input)} file(s)")

    # Plan work with resume checks
    work_items = []  # list of (h5_path, sample_name, out_path, expected_rows)
    skipped = 0
    reprocess_incomplete = 0
    zero_autosomes = 0

    for h5_path, sample_name in file_sample_pairs:
        expected = expected_rows_for_sample(h5_path, sample_name)
        legacy, disamb = out_paths_for(h5_path, sample_name, args.output)

        if expected <= 0:
            print(f"  SKIP [{sample_name}] 0 autosomal exons (or invalid group)")
            zero_autosomes += 1
            continue

        done = False
        for cand in (legacy, disamb):
            if os.path.exists(cand):
                rows = completed_rows_in_csv(cand)
                if rows >= expected:
                    print(f"  SKIP [{sample_name}] complete ({rows} rows) -> {cand}")
                    skipped += 1
                    done = True
                    break
                else:
                    print(f"  INCOMPLETE [{sample_name}] {rows}/{expected} rows -> will reprocess")
                    reprocess_incomplete += 1
                    # will overwrite using disambiguated path

        if not done:
            out_path = legacy
            work_items.append((h5_path, sample_name, out_path, expected))

    print(f"\nResume plan: {len(work_items)} to run | {skipped} already complete | "
          f"{reprocess_incomplete} will be reprocessed | {zero_autosomes} invalid/empty")

    if not work_items:
        elapsed = time.time() - t_start
        print(f"\nNothing to do. Exiting. ({elapsed:.1f}s)")
        return

    print(f"\nStarting parallel inference with {args.jobs} workers...\n")

    # Parallel scheduling
    result_queue = Queue()
    active_processes = []
    completed = 0
    sample_idx = 0
    results = []

    total_to_run = len(work_items)
    while completed < total_to_run:
        while len(active_processes) < args.jobs and sample_idx < total_to_run:
            h5_path, sample_name, out_path, expected_rows = work_items[sample_idx]
            assigned_gpu = gpu_list[sample_idx % len(gpu_list)]
            print(f"[{sample_idx+1}/{total_to_run}] Starting: {sample_name} (GPU {assigned_gpu}) -> {os.path.basename(out_path)}")

            p = Process(
                target=worker_process,
                args=(h5_path, sample_name, args.model, args.norm_file, gene2id,
                      args.batch_size, assigned_gpu, out_path, expected_rows, result_queue)
            )
            p.start()
            active_processes.append((p, sample_name, sample_idx+1))
            sample_idx += 1

        if not result_queue.empty():
            sample_name, pred_count, out_path, error = result_queue.get()
            if error:
                print(f"  ERROR [{sample_name}] {error}\n")
            else:
                print(f"  OK [{sample_name}] Wrote {pred_count:,} predictions -> {out_path}\n")
                results.append((sample_name, pred_count))
            completed += 1

            # join finished
            for i, (p, pname, _) in enumerate(active_processes):
                if pname == sample_name:
                    p.join()
                    active_processes.pop(i)
                    break

        time.sleep(0.1)

    for p, _, _ in active_processes:
        p.join()

    # Summary
    elapsed = time.time() - t_start
    total_new = len(results)
    print(f"\n{'='*60}")
    print("Testing complete!")
    print(f"  Skipped complete samples : {skipped}")
    print(f"  Reprocessed incomplete   : {reprocess_incomplete}")
    print(f"  Skipped invalid/empty    : {zero_autosomes}")
    print(f"  Newly processed          : {total_new}")
    print(f"  Total time               : {elapsed:.1f}s")
    if total_new > 0:
        print(f"  Average per processed    : {elapsed/total_new:.1f}s per sample")
    print(f"  Output directory         : {args.output}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
