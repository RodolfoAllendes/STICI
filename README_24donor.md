# STICI Imputation — 24-Donor Dataset (chr22)

## Overview

This document describes the pipeline for training and running STICI imputation on a 24-donor whole-genome sequencing dataset restricted to chromosome 22.

STICI (v1.1) is a transformer-based genotype imputation model. It is run inside a Singularity container (`~/stici.sif`) on an HPC cluster with H200 GPUs via SLURM.

---

## Data

### Reference panel

| File | Location |
|---|---|
| Phased reference (chr22) | `data/24donor/22.phased_24donor_reference.vcf.gz` |

- **3,202 samples** (diploid, phased) → 6,404 haploids used for training
- **13,178 variants** on chromosome 22
- Chromosome naming: Ensembl style (`22`, not `chr22`)

### Per-donor target files

| Directory | Description |
|---|---|
| `data/24donor/original/` | 24 per-donor VCF files, Ensembl chromosome naming (`22`), IDs annotated |

Files follow the pattern: `possorted_genome_bam_i25_donor_<N>_22_mapphased.vcf.gz`

### SNP database

| File | Location |
|---|---|
| GTC merged SNP list | `data/24donor/chr22.phased_24donor_GTC_all_merged.vcf.gz` |

Contains 15,324 rsIDs for chr22, used to annotate variant IDs in both the reference and target files.

---

## Preprocessing

### 1. Chromosome renaming

The original per-donor VCF files used UCSC-style chromosome names (`chr22`). Renaming was performed externally using `bcftools annotate` with the mapping file `~/chr2number_map.txt`. The command for future reference:

```bash
ls <input_dir>/*chr22.vcf.gz | \
    xargs -P 4 -I{} bash -c '
        f="{}"
        bcftools annotate --rename-chrs ~/chr2number_map.txt "$f" -Oz -o "${f/chr22/22}"
        bcftools index "${f/chr22/22}"
    '
```

> Note: `bcftools` must be available in the environment where this runs (not inside `stici.sif`).

### 2. Variant ID annotation

STICI matches variants between the reference and target by ID. Both files originally had `.` as the variant ID, which causes a cartesian-product merge. The script `annotate_ids.py` annotates IDs in-place: rsID from the SNP database where available, otherwise `CHROM:POS:REF:ALT`.

This step only needs to be run **once**:

```bash
singularity exec ~/stici.sif python annotate_ids.py \
    --snp-db data/24donor/chr22.phased_24donor_GTC_all_merged.vcf.gz \
    data/24donor/22.phased_24donor_reference.vcf.gz \
    data/24donor/original/
```

All 13,178 reference variants were matched to rsIDs from the SNP database.

---

## Training

**Script:** `stici_train.sh`  
**Submit:** `sbatch stici_train.sh`  
**Logs:** `logs/stici_train_<jobid>.out` / `.err`

### Parameters

| Parameter | Value | Default | Note |
|---|---|---|---|
| `--ref` | `data/24donor/22.phased_24donor_reference.vcf.gz` | — | required |
| `--save-dir` | `training_results/24donor/22_ep200` | — | required |
| `--tihp` | `true` | — | required; data is phased |
| `--epochs` | `200` | `1000` | early stopping (patience=35) will typically trigger first |
| `--batch-size-per-gpu` | `16` | `4` | tuned for H200 (141 GB); 32 caused OOM |
| `--sites-per-model` | `6144` | `6144` | default; gives 3 chunks for 13,178 variants |
| `--co` | `128` | `128` | default; chunk overlap (attention range) in SNPs |

### Hardware

- 4× NVIDIA H200 (141 GB HBM3e each)
- SLURM partition: `batch`
- 500 GB system RAM, 15 CPUs
- Multi-GPU training via TensorFlow `MirroredStrategy`
- Effective batch size: `16 × 4 GPUs = 64`

### Chunking

With 13,178 variants and `--sites-per-model 6144`, training produces **3 sub-models**:
- Chunk 1: variants 0–6,144
- Chunk 2: variants 6,144–12,288
- Chunk 3: variants 12,288–13,178

Each chunk is extended by `2 × co = 256` variants on each side for boundary context.

### Outputs

```
training_results/24donor/22_ep200/
├── models/
│   ├── w_0.ckpt      # chunk 1 model
│   ├── w_1.ckpt      # chunk 2 model
│   ├── w_2.ckpt      # chunk 3 model
│   └── chunks_info.json
└── commandline_args.json
```

---

## Imputation

**Script:** `stici_impute.sh`  
**Submit:** `sbatch stici_impute.sh`

The imputation script must point to the **same reference and save-dir** used during training. STICI reads the trained models from `save-dir` and loads training configuration from `commandline_args.json` automatically.

The script loops over all per-donor files in `TARGET_DIR` and runs imputation sequentially. Because STICI always writes to `$SAVE_DIR/out/ligated_results.vcf.gz`, the output is renamed to a donor-specific filename after each run before the next iteration overwrites it.

Key variables in `stici_impute.sh`:

```bash
REF=./data/24donor/22.phased_24donor_reference.vcf.gz
SAVE_DIR=./training_results/24donor/22_ep200
TARGET_DIR=./data/24donor/original
```

### Output

```
training_results/24donor/22_ep200/out/
├── possorted_genome_bam_i25_donor_0_22_mapphased_imputed.vcf.gz
├── possorted_genome_bam_i25_donor_1_22_mapphased_imputed.vcf.gz
├── ...
└── possorted_genome_bam_i25_donor_23_22_mapphased_imputed.vcf.gz
```

Each file contains imputed genotypes in `GT:GP:DS` format.

---

## Evaluation

**Script:** `impute_stats.py`

Computes per-variant r² and hard-call concordance against a ground-truth VCF.

```bash
singularity exec ~/stici.sif python impute_stats.py \
    --imputed training_results/24donor/22_ep200/out/ligated_results.vcf.gz \
    --truth <ground_truth.vcf.gz>
```

### Metrics reported

- Mean and median r² across all variants
- Fraction of variants with r² ≥ 0.3 / 0.5 / 0.8 / 0.9
- Mean r² and concordance stratified by MAF bins: [0.01, 0.05), [0.05, 0.10), [0.10, 0.50)
