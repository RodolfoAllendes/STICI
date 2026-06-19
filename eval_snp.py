"""
Evaluate STICI imputation quality for trained chunks.

Runs each trained chunk model against the test samples and computes
per-variant imputation r² vs ground truth. Skips chunks whose .keras
file does not exist yet.

Usage:
    python eval_snp.py
    python eval_snp.py --which-chunk 1
"""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Import STICI (filename has dots, so standard import won't work)
# ------------------------------------------------------------------
_spec = importlib.util.spec_from_file_location('STICI', 'STICI_V1.1.py')
_mod  = importlib.util.module_from_spec(_spec)
sys.modules['STICI'] = _mod
_spec.loader.exec_module(_mod)

from STICI import DataReader, get_test_dataset, custom_objects

import tensorflow as tf
import keras.backend as K

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
ROOT_DIR  = '/mnt/storage4/rallendes/STICI'
DATA_DIR  = f'{ROOT_DIR}/data/STI_benchmark_datasets'
SAVE_DIR  = f'{ROOT_DIR}/test_results/snp'

REF_VCF   = f'{DATA_DIR}/beadchip_reference_all_minaf_05_snps_hwe_1e-2_filtered_train.vcf.gz'
TEST_VCF  = f'{DATA_DIR}/test_data_beadchip_hwe_filtered.vcf.gz'
TRUE_VCF  = f'{DATA_DIR}/test_true_data_beadchip_hwe_filtered.vcf.gz'


def pearson_r2_per_variant(true_dosage: np.ndarray, pred_dosage: np.ndarray) -> np.ndarray:
    """
    Compute per-variant Pearson r² between true and imputed dosage.
    Both arrays: shape (n_samples, n_variants).
    """
    t = true_dosage - true_dosage.mean(axis=0, keepdims=True)
    p = pred_dosage - pred_dosage.mean(axis=0, keepdims=True)
    num   = (t * p).sum(axis=0)
    denom = np.sqrt((t**2).sum(axis=0) * (p**2).sum(axis=0)) + 1e-12
    return (num / denom) ** 2


def load_training_args():
    path = f'{SAVE_DIR}/commandline_args.json'
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--which-chunk', type=int, default=-1,
                        help='Evaluate a single chunk (1-indexed); -1 = all trained')
    parser.add_argument('--batch-size',  type=int, default=8)
    args = parser.parse_args()

    # Match chunk/overlap params to training run
    training_args = load_training_args()
    sites_per_model = training_args.get('sites_per_model', 10240)
    co              = training_args.get('co', 64)
    tihp            = training_args.get('tihp', True)

    # ------------------------------------------------------------------
    # Load reference + test data
    # ------------------------------------------------------------------
    print('Loading reference panel...')
    dr = DataReader()
    dr.assign_training_set(
        file_path=REF_VCF,
        target_is_gonna_be_phased_or_haps=tihp,
        variants_as_columns=False,
        delimiter=None,
        file_format='infer',
        first_column_is_index=False,
        comments='##',
    )

    print('Loading test (sparse) data...')
    dr.assign_test_set(
        file_path=TEST_VCF,
        variants_as_columns=False,
        delimiter=None,
        file_format='infer',
        first_column_is_index=False,
        comments='##',
    )

    print('Loading ground truth...')
    dr_true = DataReader()
    dr_true.assign_training_set(
        file_path=TRUE_VCF,
        target_is_gonna_be_phased_or_haps=tihp,
        variants_as_columns=False,
        delimiter=None,
        file_format='infer',
        first_column_is_index=False,
        comments='##',
    )

    # ------------------------------------------------------------------
    # Per-chunk evaluation
    # ------------------------------------------------------------------
    break_points = list(np.arange(0, dr.VARIANT_COUNT, sites_per_model)) + [dr.VARIANT_COUNT]
    n_chunks = len(break_points) - 1
    print(f'\n{n_chunks} total chunks, sites_per_model={sites_per_model}\n')

    all_r2     = []
    all_labels = []   # variant index

    for w in range(n_chunks):
        chunk_num = w + 1
        if args.which_chunk != -1 and chunk_num != args.which_chunk:
            continue

        model_path = f'{SAVE_DIR}/models/w_{w}.keras'
        if not os.path.exists(model_path):
            print(f'Chunk {chunk_num}/{n_chunks}: no model found at {model_path}, skipping.')
            continue

        print(f'Chunk {chunk_num}/{n_chunks}: running imputation...')

        final_start = max(0, break_points[w] - 2 * co)
        final_end   = min(dr.VARIANT_COUNT, break_points[w + 1] + 2 * co)

        test_np = dr.get_target_set(final_start, final_end).astype(np.int32)
        steps   = int(np.ceil(len(test_np) / args.batch_size))

        K.clear_session()
        strategy   = tf.distribute.get_strategy()
        test_ds    = get_test_dataset(test_np, args.batch_size, depth=dr.SEQ_DEPTH, strategy=strategy)

        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        predict_onehot = model.predict(test_ds, steps=steps, verbose=1)  # (n_haploids, n_variants, n_alleles)

        # Convert haploid predictions → diploid dosage (expected alt allele count per sample)
        # predict_onehot shape: (n_haploids, core_variants, n_alleles)
        # alt dosage per haploid = sum of P(alt alleles) = sum over alleles 1..end
        pred_hap_dosage = predict_onehot[:, :, 1:].sum(axis=-1)  # (n_haploids, core_variants)

        # Pair haploids back into diploid samples (haploids are interleaved: 0+1, 2+3, ...)
        n_hap, n_var = pred_hap_dosage.shape
        n_samples    = n_hap // 2
        pred_diploid = pred_hap_dosage[:2*n_samples:2] + pred_hap_dosage[1:2*n_samples:2]  # (n_samples, n_vars)

        # Ground truth diploid dosage for same positions
        true_ref_set = dr_true.get_ref_set(break_points[w], break_points[w + 1]).astype(np.float32)
        # true_ref_set shape: (n_samples*2_haploids, n_variants) — same pairing
        n_true_hap = true_ref_set.shape[0]
        n_true_sam = n_true_hap // 2
        true_diploid = true_ref_set[:2*n_true_sam:2] + true_ref_set[1:2*n_true_sam:2]

        # Align sample count (test may have fewer samples than ref)
        n_eval = min(pred_diploid.shape[0], true_diploid.shape[0])
        r2 = pearson_r2_per_variant(true_diploid[:n_eval], pred_diploid[:n_eval])

        print(f'  Chunk {chunk_num}: {n_var} variants, mean r²={r2.mean():.4f}, '
              f'median r²={np.median(r2):.4f}, '
              f'r²≥0.8: {(r2>=0.8).mean()*100:.1f}%')

        all_r2.extend(r2.tolist())
        all_labels.extend(range(break_points[w], break_points[w + 1]))

    if not all_r2:
        print('No trained chunks found to evaluate.')
        return

    # ------------------------------------------------------------------
    # Summary and plot
    # ------------------------------------------------------------------
    all_r2 = np.array(all_r2)
    print(f'\n=== Overall ({len(all_r2)} variants) ===')
    print(f'  Mean r²   : {all_r2.mean():.4f}')
    print(f'  Median r² : {np.median(all_r2):.4f}')
    for thresh in [0.5, 0.8, 0.9]:
        print(f'  r² ≥ {thresh}: {(all_r2 >= thresh).mean()*100:.1f}%')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(all_r2, bins=50, edgecolor='black', color='steelblue')
    axes[0].axvline(all_r2.mean(), color='red', linestyle='--', label=f'mean={all_r2.mean():.3f}')
    axes[0].set_xlabel('Imputation r²')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Per-variant imputation r² distribution')
    axes[0].legend()

    axes[1].scatter(all_labels, all_r2, s=1, alpha=0.3, color='steelblue')
    axes[1].set_xlabel('Variant index')
    axes[1].set_ylabel('r²')
    axes[1].set_title('r² along the chromosome')

    plt.tight_layout()
    out_path = f'{SAVE_DIR}/imputation_r2.png'
    plt.savefig(out_path, dpi=150)
    print(f'\nPlot saved to {out_path}')
    plt.show()


if __name__ == '__main__':
    main()
