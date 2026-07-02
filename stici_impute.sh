#!/bin/bash
#SBATCH --job-name=stici_impute
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=500G
#SBATCH --cpus-per-task=15
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

mkdir -p logs

# has to be the same ref and save-dir used for training

# ── 24-donor reference ────────────────────────────────────────────────────────
#REF=./data/24donor/22.phased_24donor_reference.vcf.gz
#SAVE_DIR=./training_results/24donor/22_ep200

# ── 1000G WGS reference, MAF > 0.05 ──────────────────────────────────────────
REF=./data/1kg_wgs/1kGP_high_coverage_Illumina.22.filtered.SNV_INDEL_SV_phased_panel_maf05.vcf.gz
SAVE_DIR=./training_results/1kg_wgs/22_ep200

# directory containing per-donor target files
TARGET_DIR=./data/24donor/original

export SINGULARITYENV_LD_LIBRARY_PATH=/usr/local/cuda-12.8/targets/x86_64-linux/lib

mkdir -p "$SAVE_DIR/out"

for TARGET in "$TARGET_DIR"/*_22_mapphased.vcf.gz; do
    fname=$(basename "$TARGET" .vcf.gz)
    echo "Imputing $fname ..."

    singularity exec --nv \
        -B /usr/local/cuda-12.8:/usr/local/cuda-12.8 \
        ~/stici.sif \
        python STICI_V1.1.py \
        --mode impute \
        --ref "$REF" \
        --save-dir "$SAVE_DIR" \
        --target "$TARGET" \
        --tihp true

    # STICI always writes to $SAVE_DIR/out/ligated_results.vcf.gz — rename per donor
    mv "$SAVE_DIR/out/ligated_results.vcf.gz" "$SAVE_DIR/out/${fname}_imputed.vcf.gz"
done

echo "All donors imputed."