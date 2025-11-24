import h5py
import numpy as np
import os
from tqdm import tqdm
import argparse

def filter_smn_region(input_h5, output_h5, start_coord, end_coord, chrom_num=5):
    
    print(f"Filtering {input_h5}...")
    print(f"Target Region: Chr{chrom_num}:{start_coord}-{end_coord}")
    
    with h5py.File(input_h5, 'r') as source, h5py.File(output_h5, 'w') as dest:
        
        # Iterate over every sample in the source file
        for sample_name in tqdm(source.keys(), desc="Processing Samples"):
            

            meta_data = source[sample_name]['meta'][:]

            
            is_chr5 = (meta_data[:, 2] == chrom_num)
            is_after_start = (meta_data[:, 0] >= start_coord)
            is_before_end = (meta_data[:, 1] <= end_coord)

            mask = is_chr5 & is_after_start & is_before_end
        
            if np.sum(mask) == 0:
                pass 


            indices = np.where(mask)[0]
            grp = dest.create_group(sample_name)
            
            raw_signals = source[sample_name]['signals'][:]
            filtered_signals = raw_signals[indices]
            
            grp.create_dataset('signals', data=filtered_signals, compression="gzip", chunks=True)

            filtered_meta = meta_data[indices]
            grp.create_dataset('meta', data=filtered_meta, compression="gzip", chunks=True)
            
            raw_cn = source[sample_name]['copy_number'][:]
            filtered_cn = raw_cn[indices]
            grp.create_dataset('copy_number', data=filtered_cn, compression="gzip", chunks=True)

    print(f"Done! Filtered data saved to: {output_h5}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to the big H5 file")
    parser.add_argument("--output", required=True, help="Path to save the SMN-only H5 file")
    
    parser.add_argument("--start", type=int, default=69000000, help="Start coordinate")
    parser.add_argument("--end", type=int, default=71000000, help="End coordinate")
    
    args = parser.parse_args()
    
    filter_smn_region(args.input, args.output, args.start, args.end)