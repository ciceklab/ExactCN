#!/usr/bin/env python3
import os
import glob
import logging
import pickle

import numpy as np
import h5py
from tqdm import tqdm

def compress_to_h5(npy_dir, out_h5, compression, level):
    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)s] %(message)s')
    npy_files = sorted(glob.glob(os.path.join(npy_dir, '*.npy')))
    with h5py.File(out_h5, 'a') as h5f:
        for fn in tqdm(npy_files, desc="Samples"):
            sample = os.path.splitext(os.path.basename(fn))[0]

            # if already in HDF5, skip
            if sample in h5f:
                logging.info(f"Skipping {sample}, already in HDF5")
                continue

            # otherwise, load & write
            try:
                data = np.load(fn, allow_pickle=True).item()
            except (EOFError, pickle.UnpicklingError) as e:
                logging.warning(f"Skipping {sample!r}, load failed: {e}")
                continue

            grp = h5f.create_group(sample)
            grp.create_dataset(
                'signals',
                data=data['signals'],
                compression=compression,
                compression_opts=level,
                chunks=True
            )
            grp.create_dataset(
                'meta',
                data=data['meta'],
                compression=compression,
                compression_opts=level,
                chunks=True
            )
            grp.create_dataset(
                'copy_number',
                data=data['copy_number'],
                compression=compression,
                compression_opts=level,
                chunks=True
            )

            # flush to disk
            h5f.flush()

    logging.info("Done writing HDF5.")

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(
        description="Compress per-sample .npy files into one HDF5 without deleting originals"
    )
    p.add_argument('npy_dir', help="Directory containing per-sample .npy files")
    p.add_argument('out_h5', help="Path to output HDF5 file")
    p.add_argument('--compression', default='gzip',
                   help="Compression algorithm (gzip, lzf, etc.)")
    p.add_argument('--level', type=int, default=4,
                   help="Compression level (e.g. 1–9 for gzip)")
    args = p.parse_args()
    compress_to_h5(args.npy_dir, args.out_h5, args.compression, args.level)
