#!/usr/bin/env python3
"""Merge per-donor imputed VCFs into one multi-sample VCF.

Each sample is renamed to its chip ID (parsed from the filename) in the output.

Usage:
    python merge_imputed.py --input-dir <dir> --output <merged.vcf.gz>
"""

import argparse
import glob
import gzip
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument("--input-dir", required=True, help="Directory with per-donor imputed VCFs")
parser.add_argument("--output",    required=True, help="Output merged VCF.gz path")
args = parser.parse_args()


def chip_id(path):
    m = re.search(r"(\d+_R\d+C\d+)", os.path.basename(path))
    if not m:
        raise ValueError(f"Could not parse chip ID from filename: {path}")
    return m.group(1)


files = sorted(glob.glob(os.path.join(args.input_dir, "*.vcf.gz")), key=chip_id)
print(f"Found {len(files)} files")
for f in files:
    print(f"  {chip_id(f)} <- {os.path.basename(f)}")

# read all files into memory
print("\nReading files ...", flush=True)
meta_lines = []
all_data = []

for file_idx, path in enumerate(files):
    rows = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("##"):
                if file_idx == 0:
                    meta_lines.append(line)
            elif line.startswith("#CHROM"):
                pass
            else:
                rows.append(line.split("\t"))
    all_data.append(rows)
    print(f"  {chip_id(path)}: {len(rows)} variants", flush=True)

n_donors   = len(files)
n_variants = len(all_data[0])
for i, rows in enumerate(all_data):
    if len(rows) != n_variants:
        raise ValueError(f"{chip_id(files[i])} has {len(rows)} variants, expected {n_variants}")

sample_names = [chip_id(f) for f in files]

print(f"\nWriting {n_variants} variants x {n_donors} donors -> {args.output}", flush=True)
os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

with gzip.open(args.output, "wt") as out:
    for line in meta_lines:
        out.write(line + "\n")
    out.write("\t".join(
        ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"]
        + sample_names
    ) + "\n")
    for var_idx in range(n_variants):
        fixed   = all_data[0][var_idx][:9]
        samples = [all_data[d][var_idx][9] for d in range(n_donors)]
        out.write("\t".join(fixed + samples) + "\n")

print("Done.")
