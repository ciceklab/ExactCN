#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, argparse, time
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
from torch.utils.data import Dataset, DataLoader, RandomSampler
from performer_pytorch import Performer
from einops import rearrange
from sklearn.metrics import precision_score, recall_score, r2_score, mean_squared_error
from scipy.stats import pearsonr

# Constants / Hyperparams
EXON_SIZE   = 1000
CHANNELS    = 4
PAD_VAL     = -1.0
AUTOSOMES   = set(range(1,23))
D_MODEL     = 512
DEPTH       = 5
HEADS       = 8
MAX_EXON_ID = 128

# Thresholds for binning the regression output into classes
NUM_CN_CLASSES = 3
DEL_CUTOFF = 1.60
DUP_CUTOFF = 2.20

# Regression loss weights (now all = 1)
HUBER_BETA     = 2.0
HUBER_W_NOCALL = 1
HUBER_W_DEL    = 1
HUBER_W_DUP    = 1

# Cross-entropy from regressor (soft binning)
# Temperature controls the softness (in CN units); smaller => sharper.
CE_TEMP        = 0.1
CLASS_WEIGHTS  = torch.tensor([2,8,5], dtype=torch.float32) # [DEL, NO-CALL, DUP]
CE_LOSS_SCALE  = 1  # weight of CE term relative to Huber

UNK_GENE_ID = 0
UNK_EXON_ID = 0

GLOBAL_MEAN = GLOBAL_STD = None
global_step = 0

# CLI / Env
def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune CNV regressor (regression-only + soft CE)")
    p.add_argument("-i","--input", required=True, help="Annotated HDF5 (fine-tune dataset)")
    p.add_argument("-n","--norm_file", required=True, help="Channel means/stds CSV")
    p.add_argument("-o","--output", required=True, help="Output directory for fine-tuned model")
    p.add_argument("-bs","--batch_size", type=int, required=True)
    p.add_argument("-e","--epochs", type=int, required=True)
    p.add_argument("-lr","--learning_rate", type=float, required=True)
    p.add_argument("-g","--gpu", default="")
    p.add_argument("--pretrained_dir", default=None, help="Directory containing pretrained model_latest.pt and gene_vocab.txt for fine-tuning")
    return p.parse_args()

def setup_environment(outdir, gpu):
    os.makedirs(outdir, exist_ok=True)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Normalization
def load_normalization(norm_file: str, device: torch.device):
    # load normalization from header or raw 8 floats
    try:
        rec = np.genfromtxt(norm_file, delimiter=',', names=True, dtype=float, max_rows=1)
        fields = [
            "signal_mean_A","signal_std_A",
            "signal_mean_T","signal_std_T",
            "signal_mean_C","signal_std_C",
            "signal_mean_G","signal_std_G",
        ]
        if not all(f in rec.dtype.names for f in fields):
            raise ValueError
        means = np.array([rec["signal_mean_A"], rec["signal_mean_T"],
                          rec["signal_mean_C"], rec["signal_mean_G"]], dtype=np.float32)
        stds  = np.array([rec["signal_std_A"], rec["signal_std_T"],
                          rec["signal_std_C"], rec["signal_std_G"]], dtype=np.float32)
    except Exception:
        arr = np.loadtxt(norm_file, delimiter=',').astype(np.float32).ravel()
        if arr.size < 8:
            raise RuntimeError(f"{norm_file} must contain 8 numbers")
        means = np.array([arr[0], arr[2], arr[4], arr[6]], dtype=np.float32)
        stds  = np.array([arr[1], arr[3], arr[5], arr[7]], dtype=np.float32)
    stds = np.clip(stds, 1e-6, None)
    global GLOBAL_MEAN, GLOBAL_STD
    GLOBAL_MEAN = torch.tensor(means, device=device)
    GLOBAL_STD  = torch.tensor(stds,  device=device)

    print("Loaded normalization (A,T,C,G):")
    for c, m, s in zip("ATCG", means, stds):
        print(f"  {c}: mean={m:.6f}, std={s:.6f}")

def normalize_signals(sig: torch.Tensor) -> torch.Tensor:
    mask = (sig != PAD_VAL)
    sig_n = sig.clone()
    mu = GLOBAL_MEAN.view(1, CHANNELS, 1)
    sd = GLOBAL_STD.view(1, CHANNELS, 1)
    sig_n[mask] = ((sig_n - mu)/sd)[mask]
    sig_n[~mask] = PAD_VAL
    return sig_n

# Dataset
def build_gene_vocab(h5: h5py.File) -> Tuple[List[str], Dict[str,int]]:
    genes = []
    if "exon_annotation" in h5 and "gene_name" in h5["exon_annotation"]:
        genes = list(np.unique(h5["exon_annotation"]["gene_name"][:].astype(str)))
    else:
        for k in sorted(h5.keys()):  # deterministic
            if isinstance(h5[k], h5py.Group) and "gene_name" in h5[k]:
                genes.extend(
                    x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x)
                    for x in h5[k]["gene_name"][:]
                )
    genes = [g for g in genes if g != "NA"]
    vocab = ["NA"] + sorted(set(genes))
    return vocab, {g:i for i,g in enumerate(vocab)}

class HDF5ExonDataset(Dataset):
    def __init__(self,
                 h5_path: str,
                 augment: bool = True,
                 gene_vocab: List[str] = None,
                 gene2id: Dict[str,int] = None):
        self.h5_path = h5_path
        self.index_map = []
        self.tri_labels = []
        self.by_tri_class = {0:[],1:[],2:[]}

        with h5py.File(h5_path,'r') as f:
            # If external vocab is provided (fine-tuning), use it;
            # otherwise, build from this HDF5.
            if gene_vocab is None or gene2id is None:
                self.gene_vocab, self.gene2id = build_gene_vocab(f)
            else:
                self.gene_vocab = gene_vocab
                self.gene2id   = gene2id

            # deterministic traversal order
            for s in sorted(f.keys()):
                if not isinstance(f[s], h5py.Group):
                    continue
                grp = f[s]
                if "meta" not in grp or "copy_number" not in grp:
                    continue
                meta = grp["meta"][:]
                cn   = grp["copy_number"][:]
                if meta.ndim != 2 or meta.shape[1] < 3:
                    continue
                idxs = np.where(np.isin(meta[:,2], list(AUTOSOMES)))[0]
                for i in idxs:
                    self.index_map.append((s, int(i)))
                    cn_val = float(cn[i])
                    self.tri_labels.append(
                        0 if cn_val <= DEL_CUTOFF else (2 if cn_val >= DUP_CUTOFF else 1)
                    )

        for j, t in enumerate(self.tri_labels):
            self.by_tri_class[t].append(j)

    def __len__(self): 
        return len(self.index_map)

    def _map_gene(self, g): 
        return self.gene2id.get(g, UNK_GENE_ID)

    def _map_exon(self, x): 
        return UNK_EXON_ID if x<=0 else min(int(x), MAX_EXON_ID)

    def __getitem__(self, idx):
        s,e = self.index_map[idx]
        with h5py.File(self.h5_path,'r') as f:
            grp = f[s]
            sig  = grp['signals'][e].astype(np.float32)
            meta = grp['meta'][e].astype(np.int32)  # [start,end,chr,...]
            cn   = float(grp['copy_number'][e])
            gene_name = grp['gene_name'][e]
            gname = gene_name.decode() if isinstance(gene_name, (bytes, np.bytes_)) else str(gene_name)
            exid  = int(grp['exon_id'][e]) if 'exon_id' in grp else -1
        gene_id = self._map_gene(gname)
        exon_id = self._map_exon(exid)
        return (torch.from_numpy(sig),
                torch.from_numpy(meta),
                torch.tensor(gene_id, dtype=torch.long),
                torch.tensor(exon_id, dtype=torch.long),
                torch.tensor(cn, dtype=torch.float32),
                self.tri_labels[idx], idx)

# Model
class LocalSinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len=EXON_SIZE):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0,d_model,2).float() * (-np.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos*div)
        pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer('pe', pe)
    def forward(self, B): 
        return self.pe.unsqueeze(0).expand(B,-1,-1)

class CNNTokenizer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.conv1 = nn.Conv1d(CHANNELS, 64,  kernel_size=9, padding=4)
        self.conv2 = nn.Conv1d(64,      128, kernel_size=9, padding=4)
        self.conv3 = nn.Conv1d(128,     d_model, kernel_size=1)
        self.act   = nn.GELU()
        self.norm_ch  = nn.GroupNorm(num_groups=32, num_channels=d_model) # on (B, D, N)
        self.norm_tok = nn.LayerNorm(d_model) # on (B, N, D)

    def forward(self, x, mask):
        x = x.clone()
        x[~torch.isfinite(x)] = 0.0
        x[x == PAD_VAL] = 0.0
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x) # (B, D, N)
        x = self.norm_ch(x) # GroupNorm on channels
        x = rearrange(x, 'b d n -> b n d') # (B, N, D)
        x = self.norm_tok(x)
        x = x * mask.unsqueeze(-1).float()
        return x

class CNVRegressor(nn.Module):
    def __init__(self, n_genes):
        super().__init__()
        self.tok = CNNTokenizer(D_MODEL)
        self.emb_chr = nn.Embedding(23, D_MODEL) # indices 0..22
        self.emb_gene = nn.Embedding(n_genes+1, D_MODEL) # 0 = "NA"/UNK
        self.emb_exon = nn.Embedding(MAX_EXON_ID+1, D_MODEL)
        self.pos = LocalSinusoidalPositionalEmbedding(D_MODEL)
        self.cls = nn.Parameter(torch.randn(1,1,D_MODEL))
        self.backbone = Performer(dim=D_MODEL, depth=DEPTH, heads=HEADS,
                                  ff_dropout=0.1, attn_dropout=0.1)
        self.reg_head = nn.Sequential(
            nn.LayerNorm(D_MODEL),
            nn.Linear(D_MODEL,D_MODEL),
            nn.GELU(),
            nn.Linear(D_MODEL,1)
        )

    def forward(self, sig_n, meta, gene_id, exon_id):
        B = sig_n.size(0)
        valid_mask = torch.any(sig_n!=PAD_VAL, dim=1)
        tok = self.tok(sig_n, valid_mask)

        e_chr  = self.emb_chr(meta[:,2].long().clamp(min=0, max=22)).unsqueeze(1)
        e_gene = self.emb_gene(gene_id.clamp(min=0)).unsqueeze(1)
        e_exon = self.emb_exon(exon_id.clamp(min=0, max=MAX_EXON_ID)).unsqueeze(1)
        e_ctx = e_chr + e_gene + e_exon

        x = tok + self.pos(B) + e_ctx
        cls_tok = self.cls.expand(B,1,-1) + e_ctx
        seq = torch.cat([cls_tok,x], dim=1)
        mask = torch.cat([torch.ones(B,1,dtype=torch.bool,device=seq.device), valid_mask], dim=1)

        enc = self.backbone(seq, mask=mask)
        cls_enc = enc[:,0]
        pred_reg = self.reg_head(cls_enc).squeeze(1)  # (B,)
        return pred_reg

# Loss / Metrics
def weighted_huber_simple(
    pred, target,
    beta: float = HUBER_BETA,
    weight_nocall: float = HUBER_W_NOCALL,
    weight_del: float = HUBER_W_DEL,
    weight_dup: float = HUBER_W_DUP
):
    base = F.smooth_l1_loss(pred, target, beta=beta, reduction='none')
    w = torch.ones_like(target)
    w[target <= DEL_CUTOFF] = weight_del
    w[target >= DUP_CUTOFF] = weight_dup
    nocall_mask = (target > DEL_CUTOFF) & (target < DUP_CUTOFF)
    w[nocall_mask] = weight_nocall
    return (w * base).mean()

def soft_triclass_probs_from_scalar(pred_reg: torch.Tensor, temp: float = CE_TEMP):
    t = max(1e-6, float(temp))
    s_low  = torch.sigmoid((DEL_CUTOFF - pred_reg) / t) # below DEL
    s_high = torch.sigmoid((pred_reg - DUP_CUTOFF) / t) # above DUP
    p_del = s_low
    p_dup = s_high
    p_mid = (1.0 - s_low) * (1.0 - s_high)
    probs = torch.stack([p_del, p_mid, p_dup], dim=-1) # (B,3)
    probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-12)
    return probs

def ce_from_regressor(pred_reg: torch.Tensor, y_class: torch.Tensor, class_weights: torch.Tensor, temp: float = CE_TEMP):
    probs = soft_triclass_probs_from_scalar(pred_reg, temp=temp) # (B,3)
    logp  = torch.log(probs.clamp_min(1e-12)) # (B,3)
    return F.nll_loss(logp, y_class, weight=class_weights, reduction='mean')

def cn_to_triple(y, low=DEL_CUTOFF, high=DUP_CUTOFF):
    return np.where(y<=low, 0, np.where(y>=high, 2, 1))

def compute_cumulative_metrics(preds, trues):
    p = np.array(preds,dtype=np.float64)
    t = np.array(trues,dtype=np.float64)
    mae = float(np.mean(np.abs(p-t))) if p.size else 0.0
    mse = float(mean_squared_error(t,p)) if p.size else 0.0
    r2  = float(r2_score(t,p)) if p.size else 0.0
    corr= float(pearsonr(t,p)[0]) if p.size>1 else 0.0

    p_cls = cn_to_triple(p); t_cls= cn_to_triple(t)
    prec_del = precision_score((t_cls==0),(p_cls==0),zero_division=0)
    rec_del  = recall_score((t_cls==0),(p_cls==0),zero_division=0)
    prec_noc = precision_score((t_cls==1),(p_cls==1),zero_division=0)
    rec_noc  = recall_score((t_cls==1),(p_cls==1),zero_division=0)
    prec_dup = precision_score((t_cls==2),(p_cls==2),zero_division=0)
    rec_dup  = recall_score((t_cls==2),(p_cls==2),zero_division=0)
    prec_macro = precision_score(t_cls,p_cls,average='macro',zero_division=0)
    rec_macro  = recall_score(t_cls,p_cls,average='macro',zero_division=0)

    return {'mae':mae,'mse':mse,'r2':r2,'pearson_r':corr,
            'prec_del':prec_del,'rec_del':rec_del,
            'prec_nocall':prec_noc,'rec_nocall':rec_noc,
            'prec_dup':prec_dup,'rec_dup':rec_dup,
            'prec_macro':prec_macro,'rec_macro':rec_macro}

# DP-safe load/save helpers (not strictly needed, but kept)
def _state_for_current_wrap(state_dict: Dict[str, torch.Tensor], model: nn.Module) -> Dict[str, torch.Tensor]:
    want_module = isinstance(model, nn.DataParallel)
    keys = list(state_dict.keys())
    if not keys:
        return state_dict
    has_module = keys[0].startswith("module.")
    if want_module and not has_module:
        return {f"module.{k}": v for k,v in state_dict.items()}
    if not want_module and has_module:
        return {k.replace("module.","",1): v for k,v in state_dict.items()}
    return state_dict

# Train loop (one epoch)
def train_epoch(epoch, loader, model, optim, device, dataset_size, output_dir, dataset):
    global global_step
    model.train()
    all_preds, all_trues = [], []
    acc_loss = acc_cnt = 0
    processed, next_ckpt = 0, 40 # smaller interval for small fine-tune set
    seen = set()
    latest_path = os.path.join(output_dir, "model_latest.pt")

    class_weights = CLASS_WEIGHTS.to(device)

    for batch in loader:
        sig, meta, gid, exid, true_cn, tri, idxs = batch
        sig   = sig.to(device)
        meta  = meta.to(device)
        gid   = gid.to(device)
        exid  = exid.to(device)
        true_cn = true_cn.to(device)
        tri = tri.to(device).long()

        seen.update(idxs.tolist())
        sig_n = normalize_signals(sig)

        pred_reg = model(sig_n, meta, gid, exid)  # (B,)

        # Losses
        loss_reg = weighted_huber_simple(
            pred_reg, true_cn,
            beta=HUBER_BETA,
            weight_nocall=HUBER_W_NOCALL,
            weight_del=HUBER_W_DEL,
            weight_dup=HUBER_W_DUP
        )
        loss_ce = ce_from_regressor(pred_reg, tri, class_weights, temp=CE_TEMP)

        loss = loss_reg + CE_LOSS_SCALE * loss_ce

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        global_step += 1

        bs = true_cn.size(0)
        processed += bs
        acc_loss += loss.item() * bs
        acc_cnt  += bs
        all_preds.extend(pred_reg.detach().cpu().tolist())
        all_trues.extend(true_cn.detach().cpu().tolist())

        # periodic print + overwrite save
        if processed >= next_ckpt:
            cm = compute_cumulative_metrics(all_preds, all_trues)
            pct = processed / dataset_size * 100.0
            uniq_pct_ep   = 100.0 * len(seen) / max(processed, 1)
            uniq_pct_all  = 100.0 * len(seen) / max(dataset_size, 1)

            print(f"[Epoch {epoch:02d}] {processed:,}/{dataset_size:,} exons ({pct:.1f}%) | CumMAE={cm['mae']:.4f}", flush=True)

            pred_tris = cn_to_triple(np.array(all_preds))
            true_tris = cn_to_triple(np.array(all_trues))

            # predicted counts
            n_pred_del  = int((pred_tris == 0).sum())
            n_pred_nc   = int((pred_tris == 1).sum())
            n_pred_dup  = int((pred_tris == 2).sum())

            # true counts
            n_true_del  = int((true_tris == 0).sum())
            n_true_nc   = int((true_tris == 1).sum())
            n_true_dup  = int((true_tris == 2).sum())

            print(f"   Predicted class rate (reg): DEL={(pred_tris==0).mean():.3%} "
                  f"NC={(pred_tris==1).mean():.3%} DUP={(pred_tris==2).mean():.3%}")
            print(f"   Pred counts so far:  DEL={n_pred_del}  NO-CALL={n_pred_nc}  DUP={n_pred_dup}")
            print(f"   True  counts so far: DEL={n_true_del}  NO-CALL={n_true_nc}  DUP={n_true_dup}")

            print(f"   Del:    P={cm['prec_del']:.3f} R={cm['rec_del']:.3f}")
            print(f"   NoCall: P={cm['prec_nocall']:.3f} R={cm['rec_nocall']:.3f}")
            print(f"   Dup:    P={cm['prec_dup']:.3f} R={cm['rec_dup']:.3f}")
            print(f"   Balanced Acc: {(cm['rec_del'] + cm['rec_nocall'] + cm['rec_dup'])/3:.3f}")

            print(f"\n   Unique exons seen: {len(seen):,} (~{uniq_pct_ep:.1f}% of processed; ~{uniq_pct_all:.1f}% of dataset)", flush=True)
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), latest_path)
            else:
                torch.save(model.state_dict(), latest_path)
            next_ckpt += 10000

    avg_loss = acc_loss / max(acc_cnt, 1)
    cm = compute_cumulative_metrics(all_preds, all_trues)
    return avg_loss, cm['mae'], cm['mse'], cm['r2'], cm['pearson_r'], cm['prec_macro'], cm['rec_macro']

# Main
def main():
    args = parse_args()
    device = setup_environment(args.output, args.gpu)

    load_normalization(args.norm_file, device)

    gene_vocab = None
    gene2id = None

    pretrained_vocab_path = None
    if args.pretrained_dir is not None:
        pretrained_vocab_path = os.path.join(args.pretrained_dir, "gene_vocab.txt")
        if os.path.exists(pretrained_vocab_path):
            with open(pretrained_vocab_path) as f:
                gene_vocab = [line.strip() for line in f if line.strip()]
            gene2id = {g:i for i,g in enumerate(gene_vocab)}
            print(f"Loaded existing gene vocab ({len(gene_vocab)} genes) from: {pretrained_vocab_path}")
        else:
            print(f"Warning: pretrained_dir is set but no gene_vocab.txt found at {pretrained_vocab_path}. "
                  f"Will build vocab from fine-tune dataset.")

    # Build dataset (reusing vocab if available)
    if gene_vocab is not None and gene2id is not None:
        train_ds = HDF5ExonDataset(args.input, gene_vocab=gene_vocab, gene2id=gene2id)
    else:
        train_ds = HDF5ExonDataset(args.input)
        gene_vocab = train_ds.gene_vocab
        gene2id = train_ds.gene2id

    print(f"[Dataset] exons loaded (autosomes): {len(train_ds):,}")

    # Save a copy of the vocab into the output dir (for future eval/finetune)
    vocab_out = os.path.join(args.output, "gene_vocab.txt")
    if not os.path.exists(vocab_out):
        with open(vocab_out, 'w') as f:
            for gene in gene_vocab:
                f.write(f"{gene}\n")
        print(f"Saved gene vocabulary ({len(gene_vocab)} genes) to: {vocab_out}")
    else:
        print(f"Vocab file already exists at: {vocab_out}")

    # ---- Print DEL / NO-CALL / DUP counts in the fine-tune dataset ----
    class_names = {0: "DEL", 1: "NO-CALL", 2: "DUP"}
    counts = {0: 0, 1: 0, 2: 0}
    for t in train_ds.tri_labels:
        counts[t] += 1
    total = len(train_ds)
    print("[Dataset] tri-class distribution (DEL=0, NO-CALL=1, DUP=2):")
    for c in [0, 1, 2]:
        cnt = counts[c]
        pct = 100.0 * cnt / max(total, 1)
        print(f"  {class_names[c]:7s}: {cnt:4d} ({pct:5.1f}%)")

    # Simple random sampler / DataLoader (no stratification) – good for <= 400 exons
    sampler = RandomSampler(train_ds)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    epoch_total = len(train_ds)

    n_genes = len(gene_vocab) - 1
    model = CNVRegressor(n_genes).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    ckpt_in = None
    if args.pretrained_dir is not None:
        cand = os.path.join(args.pretrained_dir, "model_latest.pt")
        if os.path.exists(cand):
            ckpt_in = cand

    #resume from output dir if no explicit pretrained_dir
    if ckpt_in is None:
        cand = os.path.join(args.output, "model_latest.pt")
        if os.path.exists(cand):
            ckpt_in = cand

    if ckpt_in is not None:
        try:
            sd = torch.load(ckpt_in, map_location=device)
            target = model.module if isinstance(model, nn.DataParallel) else model
            target.load_state_dict(sd, strict=False)
            print(f"Loaded pretrained weights from {ckpt_in}")
        except Exception as e:
            print(f"Loading pretrained weights failed ({e}); starting from random init.")
    else:
        print("No pretrained checkpoint found; training from scratch.")

    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("Model architecture: Regression-only + soft-binned CE (no classification head)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_mae = float('inf')
    latest_path_out = os.path.join(args.output, "model_latest.pt")

    for epoch in range(args.epochs):
        t0 = time.time()
        loss, mae, mse, r2, corr, prec, rec = train_epoch(
            epoch, train_loader, model, optimizer, device, epoch_total, args.output, train_ds
        )
        scheduler.step()

        # overwrite-only save at epoch end (to fine-tune output dir)
        if isinstance(model, nn.DataParallel):
            torch.save(model.module.state_dict(), latest_path_out)
        else:
            torch.save(model.state_dict(), latest_path_out)

        if mae < best_mae:
            best_mae = mae
            print(f"  ↳ New best MAE (so far): {best_mae:.4f}")

        m,s = divmod(int(time.time()-t0),60)
        print(f"[Epoch {epoch:02d} End] Train: MAE={mae:.4f} MSE={mse:.4f} R²={r2:.4f} r={corr:.4f} "
              f"Prec={prec:.3f} Rec={rec:.3f} | {m}m{s}s", flush=True)

if __name__ == "__main__":
    main()
