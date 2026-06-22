"""
Evaluate STICI imputation quality for trained chunks.

Loads trained chunk models, runs imputation on test samples, and computes
per-variant Pearson r² against ground truth. Skips chunks whose .keras
file does not exist yet.

Usage:
    python eval_snp.py --model-dir training_results/snp/models/chunk1_ep25
    python eval_snp.py --model-dir training_results/snp/models/chunk1_ep25 --which-chunk 1
"""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
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
# Defaults
# ------------------------------------------------------------------
ROOT_DIR = '/mnt/storage4/rallendes/STICI'
DATA_DIR = f'{ROOT_DIR}/data/STI_benchmark_datasets'

DEFAULT_REF  = f'{DATA_DIR}/beadchip_reference_all_minaf_05_snps_hwe_1e-2_filtered_train.vcf.gz'
DEFAULT_TEST = f'{DATA_DIR}/test_data_beadchip_hwe_filtered.vcf.gz'
DEFAULT_TRUE = f'{DATA_DIR}/test_true_data_beadchip_hwe_filtered.vcf.gz'


def pearson_r2_per_variant(true_dosage: np.ndarray, pred_dosage: np.ndarray) -> np.ndarray:
    """Per-variant Pearson r² between true and imputed dosage (n_samples, n_variants)."""
    t = true_dosage - true_dosage.mean(axis=0, keepdims=True)
    p = pred_dosage - pred_dosage.mean(axis=0, keepdims=True)
    num   = (t * p).sum(axis=0)
    denom = np.sqrt((t**2).sum(axis=0) * (p**2).sum(axis=0)) + 1e-12
    return (num / denom) ** 2


def load_training_args(model_dir):
    """Look for commandline_args.json in model_dir or one level up."""
    for path in [
        os.path.join(model_dir, 'commandline_args.json'),
        os.path.join(os.path.dirname(model_dir), 'commandline_args.json'),
    ]:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description='Evaluate STICI imputation r²')
    parser.add_argument('--model-dir',   required=True,
                        help='Directory containing w_0.keras, w_1.keras, … (e.g. training_results/snp/models/chunk1_ep25)')
    parser.add_argument('--out-dir',     default=None,
                        help='Where to save plots and results (default: alongside --model-dir)')
    parser.add_argument('--ref',         default=DEFAULT_REF,  help='Reference VCF used for training')
    parser.add_argument('--test',        default=DEFAULT_TEST, help='Sparse target VCF (masked samples)')
    parser.add_argument('--truth',       default=DEFAULT_TRUE, help='Ground truth VCF (all positions)')
    parser.add_argument('--which-chunk', type=int, default=-1,
                        help='Evaluate a single chunk (1-indexed); -1 = all trained')
    parser.add_argument('--batch-size',  type=int, default=8)
    args = parser.parse_args()

    model_dir = os.path.abspath(args.model_dir)
    out_dir   = os.path.abspath(args.out_dir) if args.out_dir else os.path.join(
        os.path.dirname(model_dir), 'eval_' + os.path.basename(model_dir)
    )
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'command'), 'w') as f:
        f.write(' '.join(sys.argv) + '\n')

    # Match chunk/overlap params to the training run
    training_args   = load_training_args(model_dir)
    sites_per_model = training_args.get('sites_per_model', 10240)
    co              = training_args.get('co', 64)
    tihp            = training_args.get('tihp', True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print('Loading reference panel...')
    dr = DataReader()
    dr.assign_training_set(
        file_path=args.ref,
        target_is_gonna_be_phased_or_haps=tihp,
        variants_as_columns=False, delimiter=None,
        file_format='infer', first_column_is_index=False, comments='##',
    )

    print('Loading sparse test data...')
    dr.assign_test_set(
        file_path=args.test,
        variants_as_columns=False, delimiter=None,
        file_format='infer', first_column_is_index=False, comments='##',
    )

    print('Loading ground truth...')
    dr_true = DataReader()
    dr_true.assign_training_set(
        file_path=args.truth,
        target_is_gonna_be_phased_or_haps=tihp,
        variants_as_columns=False, delimiter=None,
        file_format='infer', first_column_is_index=False, comments='##',
    )

    # ------------------------------------------------------------------
    # Per-chunk evaluation
    # ------------------------------------------------------------------
    break_points = list(np.arange(0, dr.VARIANT_COUNT, sites_per_model)) + [dr.VARIANT_COUNT]
    n_chunks     = len(break_points) - 1
    print(f'\n{n_chunks} total chunks, sites_per_model={sites_per_model}\n')

    all_r2     = []
    all_labels = []

    for w in range(n_chunks):
        chunk_num  = w + 1
        if args.which_chunk != -1 and chunk_num != args.which_chunk:
            continue

        model_path = os.path.join(model_dir, f'w_{w}.keras')
        if not os.path.exists(model_path):
            print(f'Chunk {chunk_num}/{n_chunks}: no model at {model_path}, skipping.')
            continue

        print(f'Chunk {chunk_num}/{n_chunks}: running imputation...')

        final_start = max(0, break_points[w] - 2 * co)
        final_end   = min(dr.VARIANT_COUNT, break_points[w + 1] + 2 * co)

        test_np  = dr.get_target_set(final_start, final_end).astype(np.int32)
        steps    = int(np.ceil(len(test_np) / args.batch_size))

        K.clear_session()
        strategy = tf.distribute.get_strategy()
        test_ds  = get_test_dataset(test_np, args.batch_size, depth=dr.SEQ_DEPTH, strategy=strategy)

        model          = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        predict_onehot = model.predict(test_ds, steps=steps, verbose=1)

        # Haploid → diploid dosage
        pred_hap   = predict_onehot[:, :, 1:].sum(axis=-1)
        n_hap, n_var = pred_hap.shape
        n_sam      = n_hap // 2
        pred_dip   = pred_hap[:2*n_sam:2] + pred_hap[1:2*n_sam:2]

        # Ground truth diploid dosage for core positions
        true_ref   = dr_true.get_ref_set(break_points[w], break_points[w + 1]).astype(np.float32)
        n_true_sam = true_ref.shape[0] // 2
        true_dip   = true_ref[:2*n_true_sam:2] + true_ref[1:2*n_true_sam:2]

        n_eval = min(pred_dip.shape[0], true_dip.shape[0])
        r2     = pearson_r2_per_variant(true_dip[:n_eval], pred_dip[:n_eval])

        print(f'  Chunk {chunk_num}: {n_var} variants | '
              f'mean r²={r2.mean():.4f} | median r²={np.median(r2):.4f} | '
              f'r²≥0.8: {(r2>=0.8).mean()*100:.1f}%')

        all_r2.extend(r2.tolist())
        all_labels.extend(range(break_points[w], break_points[w + 1]))

    if not all_r2:
        print('No trained chunks found to evaluate.')
        return

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    all_r2 = np.array(all_r2)
    print(f'\n=== Overall ({len(all_r2)} variants) ===')
    print(f'  Mean r²   : {all_r2.mean():.4f}')
    print(f'  Median r² : {np.median(all_r2):.4f}')
    for thresh in [0.5, 0.8, 0.9]:
        print(f'  r² ≥ {thresh}: {(all_r2 >= thresh).mean()*100:.1f}%')

    # Save summary
    summary_path = os.path.join(out_dir, 'r2_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f'variants evaluated : {len(all_r2)}\n')
        f.write(f'mean r²            : {all_r2.mean():.4f}\n')
        f.write(f'median r²          : {np.median(all_r2):.4f}\n')
        for thresh in [0.5, 0.8, 0.9]:
            f.write(f'r² >= {thresh}       : {(all_r2 >= thresh).mean()*100:.1f}%\n')

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(all_r2, bins=50, edgecolor='black', color='steelblue')
    axes[0].axvline(all_r2.mean(), color='red', linestyle='--', label=f'mean={all_r2.mean():.3f}')
    axes[0].set_xlabel('Imputation r²')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Per-variant imputation r²')
    axes[0].legend()

    axes[1].scatter(all_labels, all_r2, s=1, alpha=0.3, color='steelblue')
    axes[1].set_xlabel('Variant index')
    axes[1].set_ylabel('r²')
    axes[1].set_title('r² along the chromosome')

    plt.tight_layout()
    plot_path = os.path.join(out_dir, 'imputation_r2.png')
    plt.savefig(plot_path, dpi=150)
    print(f'\nResults saved to {out_dir}')
    plt.show()


if __name__ == '__main__':
    main()
