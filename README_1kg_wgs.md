# STICI Imputation — 1000 Genomes WGS Reference (chr22)

## Overview

This document describes the pipeline for training and running STICI imputation using the 1000 Genomes Project (1kGP) high-coverage WGS phased panel as the reference, targeting per-donor scRNA-seq-derived genotype profiles on chromosome 22.

The motivation for using this reference over the 24-donor-specific panel (`README_24donor.md`) is coverage density. The scRNA-seq Vireo-derived per-donor VCFs contain variants exclusively in expressed gene regions (exons/UTRs). The 24-donor reference panel (13,178 variants, genome-wide) overlapped with only 12–27% of these variants (22–208 anchor SNPs per donor). The 1kGP WGS panel at MAF > 0.05 (129,953 variants) overlaps with 79–90% of target variants (112–1,278 anchor SNPs per donor), dramatically improving imputation accuracy.

STICI (v1.1) is run inside a Singularity container (`~/stici.sif`) on an HPC cluster with H200 GPUs via SLURM.

---

## Data

### Reference panel

| File | Location |
|---|---|
| 1kGP WGS chr22, MAF > 0.05 (rsID-annotated) | `data/1kg_wgs/1kGP_high_coverage_Illumina.22.filtered.SNV_INDEL_SV_phased_panel_maf05.vcf.gz` |

- **3,202 samples** (diploid, phased) → 6,404 haploids used for training
- **129,953 variants** on chromosome 22 (MAF > 0.05 filter applied)
- Source: 1000 Genomes Project high-coverage Illumina WGS, phased with Beagle 5.4
- Chromosome naming: Ensembl style (`22`, not `chr22`)

### Per-donor target files

| Directory | Description |
|---|---|
| `data/24donor/original/` | 18 per-donor VCF files, Ensembl chromosome naming (`22`), IDs annotated |

Files follow the pattern: `possorted_genome_bam_<CHIP_ID>_22_mapphased.vcf.gz`

These are per-donor genotype profiles derived from scRNA-seq BAM files after genotype-free demultiplexing with Vireo. Each file aggregates variant calls across all cells assigned to that donor cluster.

### SNP database

| File | Location |
|---|---|
| GTC merged SNP list | `data/24donor/chr22.phased_24donor_GTC_all_merged.vcf.gz` |

Used to annotate rsIDs in both the reference panel and target files.

### Ground truth (evaluation)

| File | Location |
|---|---|
| GTC chip genotypes, chr renamed | `data/24donor/22.phased_24donor_GTC_all_merged.vcf.gz` |

- 24 samples identified by chip ID (e.g. `203808740018_R05C01`)
- 15,324 variants; evaluated against the 18 donors present in the imputed output

---

## Preprocessing

### 1. Reference panel preparation (performed externally)

The original 1kGP WGS chr22 file used UCSC-style chromosome names (`chr22`) and contained 1,066,557 variants. Two steps were applied externally (requires bcftools):

```bash
# rename chr22 → 22
bcftools annotate --rename-chrs ~/chr2number_map.txt \
    1kGP_high_coverage_Illumina.chr22.filtered.SNV_INDEL_SV_phased_panel.vcf.gz \
    -Oz -o 1kGP_high_coverage_Illumina.22.filtered.SNV_INDEL_SV_phased_panel.vcf.gz

# filter to MAF > 0.05 (~22 STICI sub-models)
bcftools view -q 0.05:minor \
    1kGP_high_coverage_Illumina.22.filtered.SNV_INDEL_SV_phased_panel.vcf.gz \
    -Oz -o 1kGP_high_coverage_Illumina.22.filtered.SNV_INDEL_SV_phased_panel_maf05.vcf.gz

bcftools index 1kGP_high_coverage_Illumina.22.filtered.SNV_INDEL_SV_phased_panel_maf05.vcf.gz
```

### 2. Variant ID annotation

The 1kGP file uses positional IDs (`22:POS:REF:ALT`). The per-donor target files have a mix of rsIDs (from GTC chip overlap) and positional fallback IDs. Annotating the reference with rsIDs from the GTC database ensures STICI can match both types:

```bash
singularity exec ~/stici.sif python annotate_ids.py \
    --snp-db data/24donor/chr22.phased_24donor_GTC_all_merged.vcf.gz \
    data/1kg_wgs/1kGP_high_coverage_Illumina.22.filtered.SNV_INDEL_SV_phased_panel_maf05.vcf.gz
```

Target files were annotated in the 24-donor pipeline (`README_24donor.md`) and do not need re-annotation.

### Anchor SNP coverage

With the 1kGP WGS reference, the fraction of per-donor target variants that match a reference panel site (and thus serve as imputation anchors) is:

| Donor | Total variants | Anchor SNPs | Coverage |
|---|---|---|---|
| 201004830081_R08C01 | 1,481 | 1,278 | 86.3% |
| 201004830117_R04C01 | 339 | 274 | 80.8% |
| 201004830117_R07C01 | 142 | 112 | 78.9% |
| 203135930052_R01C01 | 168 | 136 | 81.0% |
| 203135930053_R01C01 | 198 | 179 | 90.4% |
| 203135930053_R02C01 | 833 | 694 | 83.3% |
| 203135930053_R05C01 | 1,144 | 898 | 78.5% |
| 203135930053_R06C01 | 1,012 | 866 | 85.6% |
| 203135930053_R07C01 | 177 | 145 | 81.9% |
| 203135930053_R08C01 | 171 | 140 | 81.9% |
| 203808740018_R01C01 | 1,060 | 879 | 82.9% |
| 203808740018_R02C01 | 120 | 105 | 87.5% |
| 203808740018_R05C01 | 551 | 459 | 83.3% |
| 203808740018_R07C01 | 97 | 83 | 85.6% |
| 203808740018_R08C01 | 157 | 137 | 87.3% |
| 203808740020_R02C01 | 959 | 827 | 86.2% |
| 203808740020_R03C01 | 418 | 348 | 83.3% |
| 203808740020_R07C01 | 206 | 169 | 82.0% |

For comparison, the 24-donor reference achieved only 12–27% anchor coverage (22–208 SNPs per donor).

---

## Training

**Script:** `stici_train.sh`  
**Submit:** `sbatch stici_train.sh`  
**Logs:** `logs/stici_train_<jobid>.out` / `.err`

### Parameters

| Parameter | Value | Default | Note |
|---|---|---|---|
| `--ref` | `data/1kg_wgs/1kGP_high_coverage_Illumina.22.filtered.SNV_INDEL_SV_phased_panel_maf05.vcf.gz` | — | required |
| `--save-dir` | `training_results/1kg_wgs/22_ep200` | — | required |
| `--tihp` | `true` | — | required; data is phased |
| `--epochs` | `200` | `1000` | early stopping (patience=35) will typically trigger first |
| `--batch-size-per-gpu` | `16` | `4` | tuned for H200 (141 GB) |
| `--sites-per-model` | `6144` | `6144` | default; gives ~22 sub-models for 129,953 variants |

### Hardware

- 4× NVIDIA H200 (141 GB HBM3e each)
- SLURM partition: `batch` (no wall-time limit)
- 500 GB system RAM, 15 CPUs
- Multi-GPU training via TensorFlow `MirroredStrategy` (single node)
- Effective batch size: `16 × 4 GPUs = 64`

### Chunking

With 129,953 variants and `--sites-per-model 6144`, training produces **~22 sub-models**. Training time is approximately 7× longer than the 24-donor reference run (3 sub-models).

---

## Imputation

**Script:** `stici_impute.sh`  
**Submit:** `sbatch stici_impute.sh`

The script loops over all 18 per-donor files in `TARGET_DIR` sequentially. STICI always writes to `$SAVE_DIR/out/ligated_results.vcf.gz`; each output is renamed to a donor-specific filename immediately after each run.

Key variables in `stici_impute.sh`:

```bash
REF=./data/1kg_wgs/1kGP_high_coverage_Illumina.22.filtered.SNV_INDEL_SV_phased_panel_maf05.vcf.gz
SAVE_DIR=./training_results/1kg_wgs/22_ep200
TARGET_DIR=./data/24donor/original
```

### Output

```
training_results/1kg_wgs/22_ep200/out/
├── possorted_genome_bam_201004830081_R08C01_22_mapphased_imputed.vcf.gz
├── possorted_genome_bam_201004830117_R04C01_22_mapphased_imputed.vcf.gz
├── ...
└── possorted_genome_bam_203808740020_R07C01_22_mapphased_imputed.vcf.gz
```

Each file contains imputed genotypes in `GT:GP:DS` format across all 129,953 reference panel sites.

---

## Evaluation

After imputation, merge the 18 per-donor files and compare against the GTC truth:

```bash
# merge 18 per-donor imputed VCFs into one multi-sample file
singularity exec ~/stici.sif python merge_imputed.py \
    --input-dir training_results/1kg_wgs/22_ep200/out \
    --output    training_results/1kg_wgs/22_ep200/out/merged_imputed.vcf.gz

# evaluate against GTC truth (sample matching by chip ID name)
singularity exec ~/stici.sif python impute_stats.py \
    --imputed training_results/1kg_wgs/22_ep200/out/merged_imputed.vcf.gz \
    --truth   data/24donor/22.phased_24donor_GTC_all_merged.vcf.gz \
    --output  training_results/1kg_wgs/22_ep200/out/stats.txt
```

`impute_stats.py` automatically matches the 18 imputed samples to their corresponding columns in the 24-sample GTC truth by chip ID name, and evaluates only at variants present in both files.
