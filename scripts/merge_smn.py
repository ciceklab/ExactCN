#!/usr/bin/env python3
import h5py
import numpy as np
import argparse
import sys

def decode_if_bytes(x):
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode('utf-8')
    return str(x)

def process_smn_merge(input_h5, output_h5):
    print(f"Opening input: {input_h5}")
    print(f"Writing output to: {output_h5}")

    with h5py.File(input_h5, 'r') as f_in, h5py.File(output_h5, 'w') as f_out:
        samples = list(f_in.keys())
        # Filter out non-sample keys like 'exon_annotation'
        samples = [s for s in samples if 'meta' in f_in[s] and 'signals' in f_in[s]]
        
        count = 0
        for sample in samples:
            grp_in = f_in[sample]
            
            # Load Datasets
            signals = grp_in['signals'][:]
            meta = grp_in['meta'][:]
            
            # Handle Gene Names (Bytes vs String)
            if 'gene_name' not in grp_in:
                print(f"Skipping {sample}: No gene_name found.")
                continue
                
            raw_genes = grp_in['gene_name'][:]
            gene_names = np.array([decode_if_bytes(g) for g in raw_genes])
            
            # Handle Exon IDs
            if 'exon_id' in grp_in:
                exon_ids = grp_in['exon_id'][:]
            else:
                print(f"Skipping {sample}: No exon_id found.")
                continue

            # --- FILTERING LOGIC ---
            # 1. Find indices for SMN1 Exon 8 AND SMN2 Exon 8
            # Note: We look for exact matches.
            idx_smn1 = np.where((gene_names == 'SMN1') & (exon_ids == 8))[0]
            idx_smn2 = np.where((gene_names == 'SMN2') & (exon_ids == 8))[0]

            # We need both to exist to perform a sum
            if len(idx_smn1) == 0 or len(idx_smn2) == 0:
                print(f"  [Warning] {sample}: Could not find both SMN1:Ex8 and SMN2:Ex8. Skipping.")
                continue

            # Take the first match if multiple exist (unlikely but safe)
            i1 = idx_smn1[0]
            i2 = idx_smn2[0]

            # 2. Sum the Read Depths (Signals) element by element
            # Shape is (4, 1000)
            sig_smn1 = signals[i1]
            sig_smn2 = signals[i2]
            
            summed_signal = sig_smn1 + sig_smn2

            # 3. Create Output Data
            # We keep SMN2's metadata (coordinates) as requested (SMN1 removed, SMN2 kept)
            out_signal = np.expand_dims(summed_signal, axis=0) # Shape (1, 4, 1000)
            out_meta = np.expand_dims(meta[i2], axis=0)        # Shape (1, 3)
            
            # For HDF5 string compatibility
            dt_str = h5py.string_dtype(encoding='utf-8')
            out_gene = np.array(['SMN2'], dtype=object)
            out_exon = np.array([8], dtype=np.int32)
            
            # Dummy copy_number if it exists, otherwise -1
            cn_val = -1
            if 'copy_number' in grp_in:
                # Sum CNs? Or keep SMN2? Usually for 'ExactCN' inference we use dummy -1
                cn_val = grp_in['copy_number'][i2]
            out_cn = np.array([cn_val], dtype=np.int8)

            # 4. Write to Output HDF5
            grp_out = f_out.create_group(sample)
            grp_out.create_dataset('signals', data=out_signal, compression="gzip")
            grp_out.create_dataset('meta', data=out_meta, compression="gzip")
            grp_out.create_dataset('gene_name', data=out_gene, dtype=dt_str, compression="gzip")
            grp_out.create_dataset('exon_id', data=out_exon, dtype='int32', compression="gzip")
            grp_out.create_dataset('copy_number', data=out_cn, dtype='int8')
            
            count += 1
            
        print(f"Successfully processed {count} samples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    process_smn_merge(args.input, args.output)