#!/usr/bin/env python3
"""Imputation accuracy statistics: r², concordance, by MAF bin and per donor."""

import argparse
import gzip
import math
import os
import sys
from collections import defaultdict

_p = argparse.ArgumentParser(description="Imputation r² and concordance stats")
_p.add_argument("--imputed", default="training_results/snp/bench_ep200/out/ligated_results.vcf.gz",
                help="Imputed VCF(.gz) with DS field")
_p.add_argument("--truth",   default="data/STI_benchmark_datasets/test_true_data_beadchip_hwe_filtered.vcf.gz",
                help="Ground-truth VCF(.gz) with GT field")
_p.add_argument("--output",  default=None,
                help="Output stats file (default: <imputed_dir>/stats.txt)")
_args = _p.parse_args()

IMPUTED = _args.imputed
TRUTH   = _args.truth
OUT     = _args.output or os.path.join(os.path.dirname(IMPUTED), "stats.txt")


def open_vcf(path):
    return gzip.open(path, "rt")


def parse_vcf(path, extract_fn):
    """Yield (chrom, pos, ref, alt, [values per sample]) and return sample names."""
    samples = []
    with open_vcf(path) as fh:
        for line in fh:
            if line.startswith("##"):
                continue
            if line.startswith("#"):
                samples = line.rstrip("\n").split("\t")[9:]
                continue
            cols = line.rstrip("\n").split("\t")
            chrom, pos, ref, alt = cols[0], cols[1], cols[3], cols[4]
            fmt = cols[8].split(":")
            values = [extract_fn(cols[9 + i], fmt) for i in range(len(samples))]
            yield chrom, pos, ref, alt, values
    return samples


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


def mean_v(vals):
    valid = [v for v in vals if not math.isnan(v)]
    return sum(valid) / len(valid) if valid else float("nan")


def median_v(vals):
    valid = sorted(v for v in vals if not math.isnan(v))
    if not valid:
        return float("nan")
    n = len(valid)
    return (valid[n // 2] if n % 2 else (valid[n // 2 - 1] + valid[n // 2]) / 2)


def pct_above(vals, threshold):
    valid = [v for v in vals if not math.isnan(v)]
    return 100.0 * sum(v >= threshold for v in valid) / len(valid) if valid else float("nan")


# ── load truth ────────────────────────────────────────────────────────────────
print("Reading truth GT ...", flush=True)
truth = {}
truth_samples = []
with open_vcf(TRUTH) as fh:
    for line in fh:
        if line.startswith("##"):
            continue
        if line.startswith("#"):
            truth_samples = line.rstrip("\n").split("\t")[9:]
            continue
        cols = line.rstrip("\n").split("\t")
        chrom, pos, ref, alt = cols[0], cols[1], cols[3], cols[4]
        fmt = cols[8].split(":")
        truth[(chrom, pos, ref, alt)] = [gt_to_dosage(cols[9 + i], fmt) for i in range(len(truth_samples))]
print(f"  {len(truth)} variants, {len(truth_samples)} samples", flush=True)

# ── load imputed ──────────────────────────────────────────────────────────────
print("Reading imputed DS ...", flush=True)
imputed = {}
imp_samples = []
with open_vcf(IMPUTED) as fh:
    for line in fh:
        if line.startswith("##"):
            continue
        if line.startswith("#"):
            imp_samples = line.rstrip("\n").split("\t")[9:]
            continue
        cols = line.rstrip("\n").split("\t")
        chrom, pos, ref, alt = cols[0], cols[1], cols[3], cols[4]
        fmt = cols[8].split(":")
        imputed[(chrom, pos, ref, alt)] = [ds_from_field(cols[9 + i], fmt) for i in range(len(imp_samples))]
print(f"  {len(imputed)} variants, {len(imp_samples)} samples", flush=True)

# ── match variants and samples ───────────────────────────────────────────────
common_keys = sorted(set(truth) & set(imputed))

name_overlap = set(truth_samples) & set(imp_samples)
if name_overlap:
    # match samples by name (robust: works regardless of column order)
    sample_names = sorted(name_overlap, key=imp_samples.index)
    truth_idx = [truth_samples.index(n) for n in sample_names]
    imp_idx   = [imp_samples.index(n) for n in sample_names]
    print(f"Matching {len(sample_names)} samples by name "
          f"(truth has {len(truth_samples)}, imputed has {len(imp_samples)})", flush=True)
else:
    # no name overlap: fall back to column order
    sample_names = imp_samples if imp_samples else [str(i) for i in range(len(imp_samples))]
    truth_idx = list(range(len(sample_names)))
    imp_idx   = list(range(len(sample_names)))
    print("No sample name overlap — falling back to column-order matching", flush=True)

n_samples = len(sample_names)
print(f"\nMatched variants: {len(common_keys)}  |  Samples: {n_samples}\n", flush=True)

# ── per-variant stats ─────────────────────────────────────────────────────────
r2_vals   = []
conc_vals = []
maf_r2    = defaultdict(list)
maf_conc  = defaultdict(list)

# accumulators for per-sample stats (lists of (true_ds, imp_ds) per sample)
sample_true = [[] for _ in range(n_samples)]
sample_imp  = [[] for _ in range(n_samples)]
sample_concordance = [[] for _ in range(n_samples)]

for key in common_keys:
    true_ds = [truth[key][ti] for ti in truth_idx]
    imp_ds  = [imputed[key][ii] for ii in imp_idx]

    r2 = pearson_r2(imp_ds, true_ds)
    r2_vals.append(r2)

    correct = total = 0
    for i, (td, id_) in enumerate(zip(true_ds, imp_ds)):
        if math.isnan(td) or math.isnan(id_):
            continue
        sample_true[i].append(td)
        sample_imp[i].append(id_)
        true_gt = round(td)
        imp_gt  = round(id_)
        match = int(true_gt == imp_gt)
        correct += match
        total   += 1
        sample_concordance[i].append(match)

    conc_vals.append(correct / total if total else float("nan"))

    dosages = [d for d in true_ds if not math.isnan(d)]
    if dosages:
        af  = sum(dosages) / (2 * len(dosages))
        maf = min(af, 1 - af)
        b   = maf_bin(maf)
        if b:
            if not math.isnan(r2):
                maf_r2[b].append(r2)
            maf_conc[b].append(correct / total if total else float("nan"))

# ── per-sample stats ──────────────────────────────────────────────────────────
sample_r2   = [pearson_r2(sample_imp[i], sample_true[i]) for i in range(n_samples)]
sample_conc = [mean_v(sample_concordance[i]) for i in range(n_samples)]

# ── format output ─────────────────────────────────────────────────────────────
lines = []
lines.append(f"Imputed : {IMPUTED}")
lines.append(f"Truth   : {TRUTH}")
lines.append(f"Matched : {len(common_keys)} variants  |  {n_samples} samples")
lines.append("")

lines.append(f"=== Overall r² ({len(r2_vals)} variants) ===")
lines.append(f"  Mean r²   : {mean_v(r2_vals):.4f}")
lines.append(f"  Median r² : {median_v(r2_vals):.4f}")
for thr in (0.3, 0.5, 0.8, 0.9):
    lines.append(f"  r² >= {thr}  : {pct_above(r2_vals, thr):.1f}%")
lines.append("")

lines.append("=== Concordance (hard-call) ===")
lines.append(f"  Mean      : {mean_v(conc_vals)*100:.2f}%")
lines.append(f"  Median    : {median_v(conc_vals)*100:.2f}%")
lines.append("")

lines.append("=== By MAF (from truth) ===")
for label in ["[0.01,0.05)", "[0.05,0.10)", "[0.10,0.50)"]:
    r2s  = maf_r2.get(label, [])
    cons = maf_conc.get(label, [])
    lines.append(f"  MAF {label}: n={len(r2s):5d}  mean r²={mean_v(r2s):.4f}  "
                 f"mean concordance={mean_v(cons)*100:.2f}%")
lines.append("")

lines.append("=== Per-donor stats ===")
lines.append(f"  {'Donor':<50}  {'r²':>8}  {'Concordance':>12}")
lines.append(f"  {'-'*50}  {'-'*8}  {'-'*12}")
for i, name in enumerate(sample_names):
    r2_s   = sample_r2[i]
    conc_s = sample_conc[i]
    r2_str   = f"{r2_s:.4f}"   if not math.isnan(r2_s)   else "   nan"
    conc_str = f"{conc_s*100:.2f}%" if not math.isnan(conc_s) else "    nan"
    lines.append(f"  {name:<50}  {r2_str:>8}  {conc_str:>12}")
lines.append("")
lines.append(f"  Mean per-donor r²          : {mean_v(sample_r2):.4f}")
lines.append(f"  Mean per-donor concordance : {mean_v(sample_conc)*100:.2f}%")

output = "\n".join(lines)
print(output)

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
with open(OUT, "w") as f:
    f.write(output + "\n")
print(f"\nResults saved to {OUT}", flush=True)
