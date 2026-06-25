#!/usr/bin/env python3
"""Imputation accuracy statistics: r², concordance, by MAF bin."""

import argparse
import gzip
import math
import sys
from collections import defaultdict

_p = argparse.ArgumentParser(description="Imputation r² and concordance stats")
_p.add_argument("--imputed", default="training_results/snp/bench_ep200/out/ligated_results.vcf.gz",
                help="Imputed VCF(.gz) with DS field")
_p.add_argument("--truth",   default="data/test_true_data_beadchip_hwe_filtered.vcf.gz",
                help="Ground-truth VCF(.gz) with GT field")
_args = _p.parse_args()
IMPUTED = _args.imputed
TRUTH   = _args.truth


def open_vcf(path):
    return gzip.open(path, "rt")


def parse_vcf(path, extract_fn):
    """Yield (chrom, pos, ref, alt, [values per sample]) for each variant."""
    with open_vcf(path) as fh:
        for line in fh:
            if line.startswith("##"):
                continue
            if line.startswith("#"):
                samples = line.rstrip("\n").split("\t")[9:]
                continue
            cols = line.rstrip("\n").split("\t")
            chrom, pos, _, ref, alt = cols[0], cols[1], cols[2], cols[3], cols[4]
            fmt = cols[8].split(":")
            values = [extract_fn(cols[9 + i], fmt) for i in range(len(samples))]
            yield chrom, pos, ref, alt, values


def gt_to_dosage(field, fmt):
    gt = field.split(":")[0]
    alleles = gt.replace("|", "/").split("/")
    try:
        return float(sum(int(a) for a in alleles))
    except ValueError:
        return float("nan")


def ds_from_field(field, fmt):
    parts = field.split(":")
    ds_idx = fmt.index("DS") if "DS" in fmt else None
    if ds_idx is not None and ds_idx < len(parts):
        try:
            return float(parts[ds_idx])
        except ValueError:
            return float("nan")
    return float("nan")


def pearson_r2(xs, ys):
    n = 0
    sx = sy = sxx = syy = sxy = 0.0
    for x, y in zip(xs, ys):
        if math.isnan(x) or math.isnan(y):
            continue
        n += 1
        sx += x; sy += y
        sxx += x * x; syy += y * y; sxy += x * y
    if n < 2:
        return float("nan")
    denom = math.sqrt(max(0.0, n * sxx - sx * sx) * max(0.0, n * syy - sy * sy))
    if denom == 0:
        return float("nan")
    r = (n * sxy - sx * sy) / denom
    return r * r


def maf_bin(maf):
    if maf < 0.01:
        return None
    if maf < 0.05:
        return "[0.01,0.05)"
    if maf < 0.10:
        return "[0.05,0.10)"
    return "[0.10,0.50)"


# ── load truth ───────────────────────────────────────────────────────────────
print("Reading truth GT ...", flush=True)
truth = {}
for chrom, pos, ref, alt, vals in parse_vcf(TRUTH, gt_to_dosage):
    truth[(chrom, pos, ref, alt)] = vals
print(f"  {len(truth)} variants loaded", flush=True)

# ── load imputed ─────────────────────────────────────────────────────────────
print("Reading imputed DS ...", flush=True)
imputed = {}
for chrom, pos, ref, alt, vals in parse_vcf(IMPUTED, ds_from_field):
    imputed[(chrom, pos, ref, alt)] = vals
print(f"  {len(imputed)} variants loaded", flush=True)

# ── compute per-variant r² and concordance ───────────────────────────────────
common_keys = sorted(set(truth) & set(imputed))
n_samples = len(next(iter(truth.values())))

print(f"\nMatched variants: {len(common_keys)}  |  Samples: {n_samples}\n", flush=True)

r2_vals = []
conc_vals = []
maf_r2   = defaultdict(list)
maf_conc = defaultdict(list)

for key in common_keys:
    true_ds  = truth[key]
    imp_ds   = imputed[key]

    # r²
    r2 = pearson_r2(imp_ds, true_ds)
    r2_vals.append(r2)

    # concordance (hard-call GT agreement)
    correct = total = 0
    for td, id_ in zip(true_ds, imp_ds):
        if math.isnan(td) or math.isnan(id_):
            continue
        true_gt = round(td)
        imp_gt  = round(id_)
        correct += (true_gt == imp_gt)
        total   += 1
    conc_vals.append(correct / total if total else float("nan"))

    # MAF from truth
    dosages = [d for d in true_ds if not math.isnan(d)]
    if dosages:
        af = sum(dosages) / (2 * len(dosages))
        maf = min(af, 1 - af)
        b = maf_bin(maf)
        if b:
            if not math.isnan(r2):
                maf_r2[b].append(r2)
            maf_conc[b].append(correct / total if total else float("nan"))


def pct_above(vals, threshold):
    valid = [v for v in vals if not math.isnan(v)]
    return 100.0 * sum(v >= threshold for v in valid) / len(valid) if valid else float("nan")


def mean_v(vals):
    valid = [v for v in vals if not math.isnan(v)]
    return sum(valid) / len(valid) if valid else float("nan")


def median_v(vals):
    valid = sorted(v for v in vals if not math.isnan(v))
    if not valid:
        return float("nan")
    n = len(valid)
    return (valid[n // 2] if n % 2 else (valid[n // 2 - 1] + valid[n // 2]) / 2)


# ── print results ─────────────────────────────────────────────────────────────
print(f"=== r2 ({len(r2_vals)} variants, {n_samples} samples) ===")
print(f"  Mean r2   : {mean_v(r2_vals):.4f}")
print(f"  Median r2 : {median_v(r2_vals):.4f}")
for thr in (0.3, 0.5, 0.8, 0.9):
    print(f"  r2 >= {thr} : {pct_above(r2_vals, thr):.1f}%")

print()
print(f"=== Concordance (hard-call) ===")
print(f"  Mean      : {mean_v(conc_vals)*100:.2f}%")
print(f"  Median    : {median_v(conc_vals)*100:.2f}%")

print()
print("By MAF (from truth):")
for label in ["[0.01,0.05)", "[0.05,0.10)", "[0.10,0.50)"]:
    r2s  = maf_r2.get(label, [])
    cons = maf_conc.get(label, [])
    print(f"  MAF {label}: n={len(r2s):5d}  mean r2={mean_v(r2s):.4f}  "
          f"mean concordance={mean_v(cons)*100:.2f}%")
