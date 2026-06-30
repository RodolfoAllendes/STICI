#!/usr/bin/env python
"""Annotate VCF variant IDs using a reference SNP database VCF.

For each variant, the ID is set to the rsID from the SNP DB if found,
otherwise falls back to CHROM:POS:REF:ALT. Files are overwritten in-place.

Usage:
    python annotate_ids.py --snp-db <db.vcf.gz> <file1.vcf.gz> [file2.vcf.gz ...]
"""

import argparse
import gzip
import os


def open_vcf(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")


def strip_chr(chrom):
    return chrom[3:] if chrom.startswith("chr") else chrom


def build_snp_db(snp_db_path):
    print("Building SNP lookup ...", flush=True)
    db = {}
    with open_vcf(snp_db_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            chrom, pos, rsid, ref, alt = strip_chr(cols[0]), cols[1], cols[2], cols[3], cols[4]
            if rsid != ".":
                db[(chrom, pos, ref, alt)] = rsid
    print(f"  {len(db):,} rsIDs loaded", flush=True)
    return db


def annotate_file(path, db):
    tmp = path + ".tmp.gz"
    annotated = missing = 0
    with open_vcf(path) as fin, gzip.open(tmp, "wt") as fout:
        for line in fin:
            if line.startswith("#"):
                fout.write(line)
                continue
            cols = line.rstrip("\n").split("\t")
            chrom, pos, ref, alt = strip_chr(cols[0]), cols[1], cols[3], cols[4]
            key = (chrom, pos, ref, alt)
            if key in db:
                cols[2] = db[key]
                annotated += 1
            else:
                cols[2] = f"{chrom}:{pos}:{ref}:{alt}"
                missing += 1
            fout.write("\t".join(cols) + "\n")
    os.replace(tmp, path)
    print(f"  rsID: {annotated:,}  |  fallback: {missing:,}  |  -> {path}", flush=True)


def main():
    import glob
    parser = argparse.ArgumentParser()
    parser.add_argument("--snp-db", required=True, help="VCF with rsIDs (may use chr prefix)")
    parser.add_argument("files", nargs="+", help="VCF files or directories to annotate in-place")
    args = parser.parse_args()

    # expand directories to *.vcf.gz files inside them
    paths = []
    for f in args.files:
        if os.path.isdir(f):
            paths.extend(sorted(glob.glob(os.path.join(f, "*.vcf.gz"))))
        else:
            paths.append(f)

    db = build_snp_db(args.snp_db)
    for f in paths:
        print(f"Annotating {os.path.basename(f)} ...", flush=True)
        annotate_file(f, db)
    print("Done.")


if __name__ == "__main__":
    main()
