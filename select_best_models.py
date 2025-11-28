#!/usr/bin/env python
"""
Select best models per channel based on different criteria:
1. Top models by F1 score (event-wise)
2. Best model by RMSE (with F1 as tiebreaker)
3. Compute macro and micro averages
"""
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Select best models per channel'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='anom_exp_latest',
        help='Output directory containing results'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=2,
        help='Number of top models to select per channel (default: 2)'
    )
    parser.add_argument(
        '--metric_type',
        type=str,
        default='event',
        choices=['event', 'exact'],
        help='Metric type for F1 selection (default: event)'
    )
    return parser.parse_args()


def prf_from_counts(tp, fp, fn, beta=1.0):
    """Compute precision, recall, F-beta from counts."""
    tp = float(tp)
    fp = float(fp)
    fn = float(fn)
    
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if p == 0.0 and r == 0.0:
        return p, r, 0.0
    
    beta2 = beta ** 2
    f = (1 + beta2) * p * r / (beta2 * p + r) if (p + r) > 0 else 0.0
    
    return p, r, f


def compute_macro_micro(df):
    """
    Compute macro and micro averages.
    
    Macro: Average of per-channel metrics
    Micro: Aggregate all TP/FP/FN then compute metrics
    """
    # Macro averages (average of per-channel metrics)
    macro_precision = df['precision'].mean()
    macro_recall = df['recall'].mean()
    macro_f1 = df['F1'].mean()
    macro_f05 = df['F0.5'].mean()
    
    # Micro averages (aggregate counts then compute)
    total_tp = df['tp'].sum()
    total_fp = df['fp'].sum()
    total_fn = df['fn'].sum()
    
    micro_p, micro_r, micro_f1 = prf_from_counts(total_tp, total_fp, total_fn, beta=1.0)
    _, _, micro_f05 = prf_from_counts(total_tp, total_fp, total_fn, beta=0.5)
    
    return {
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_F1': macro_f1,
        'macro_F0.5': macro_f05,
        'micro_precision': micro_p,
        'micro_recall': micro_r,
        'micro_F1': micro_f1,
        'micro_F0.5': micro_f05,
        'total_tp': int(total_tp),
        'total_fp': int(total_fp),
        'total_fn': int(total_fn),
        'num_channels': len(df)
    }


def select_top_by_f1(combined_metrics_path, top_k=2):
    """
    Select top K models per channel by F1 score.
    
    Args:
        combined_metrics_path: Path to combined metrics CSV
        top_k: Number of top models to select
    
    Returns:
        DataFrame with top models per channel
    """
    df = pd.read_csv(combined_metrics_path)
    
    # Ensure F1 is numeric
    df['F1'] = pd.to_numeric(df['F1'], errors='coerce').fillna(0.0)
    
    # Rank models per channel by F1 (descending)
    df['rank_f1'] = df.groupby('dataset')['F1'].rank(
        ascending=False,
        method='first'
    )
    
    # Select top K
    top_models = df[df['rank_f1'] <= top_k].copy()
    top_models = top_models.sort_values(['dataset', 'rank_f1', 'model'])
    
    return top_models


def select_best_by_rmse_f1(rmse_path, metrics_path):
    """
    Select best model per channel by:
    1. Lowest RMSE (primary criterion)
    2. Highest F1 (tiebreaker)
    
    Returns metrics for the best model per channel.
    """
    # Load RMSE data
    rmse_df = pd.read_csv(rmse_path)
    rmse_df['rmse_train'] = pd.to_numeric(
        rmse_df['rmse_train'],
        errors='coerce'
    )
    rmse_df = rmse_df.dropna(subset=['rmse_train'])
    
    # Load metrics data
    metrics_df = pd.read_csv(metrics_path)
    metrics_df['F1'] = pd.to_numeric(
        metrics_df['F1'],
        errors='coerce'
    ).fillna(0.0)
    
    # Merge RMSE with metrics
    merged = rmse_df.merge(
        metrics_df[['model', 'dataset', 'precision', 'recall', 'F1', 'F0.5', 'tp', 'fp', 'fn']],
        on=['model', 'channel'] if 'channel' in rmse_df.columns else ['model', 'dataset'],
        how='left',
        suffixes=('', '_metric')
    )
    
    # Handle column naming
    if 'channel' in merged.columns and 'dataset' not in merged.columns:
        merged['dataset'] = merged['channel']
    
    # Sort by RMSE (ascending) then F1 (descending)
    merged = merged.sort_values(
        ['dataset', 'rmse_train', 'F1'],
        ascending=[True, True, False]
    )
    
    # Select best model per channel (first row after sorting)
    best_models = merged.groupby('dataset', as_index=False).first()
    
    return best_models


def main():
    args = parse_args()
    
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")
    
    save_dir = output_dir
    
    print("=" * 80)
    print("SELECTING BEST MODELS PER CHANNEL")
    print("=" * 80)
    
    # ========== 1. TOP MODELS BY F1 SCORE ==========
    print(f"\n[1/2] Selecting top {args.top_k} models per channel by F1 score...")
    
    if args.metric_type == 'event':
        combined_path = save_dir / 'all_event_metrics_combined.csv'
    else:
        combined_path = save_dir / 'all_exact_metrics_combined.csv'
    
    if not combined_path.exists():
        print(f"✗ Combined metrics file not found: {combined_path}")
        print("  Run combine_metrics.py first!")
    else:
        try:
            top_models = select_top_by_f1(combined_path, top_k=args.top_k)
            
            # Save results
            output_path = save_dir / f'top{args.top_k}_models_by_f1_{args.metric_type}.csv'
            top_models.to_csv(output_path, index=False)
            
            print(f"✓ Saved: {output_path}")
            print(f"\n--- Top {args.top_k} Models per Channel (by F1) ---")
            
            # Show summary
            for dataset in sorted(top_models['dataset'].unique()):
                print(f"\n{dataset}:")
                subset = top_models[top_models['dataset'] == dataset]
                for _, row in subset.iterrows():
                    print(f"  Rank {int(row['rank_f1'])}: {row['model']:<25} "
                          f"F1={row['F1']:.4f}, F0.5={row['F0.5']:.4f}")
            
        except Exception as e:
            print(f"✗ Error selecting top models by F1: {e}")
    
    # ========== 2. BEST MODEL BY RMSE + F1 ==========
    print("\n" + "=" * 80)
    print("\n[2/2] Selecting best model per channel by RMSE (with F1 tiebreaker)...")
    
    rmse_path = save_dir / 'train_rmse_per_channel.csv'
    
    if not rmse_path.exists():
        print(f"✗ RMSE file not found: {rmse_path}")
        print("  Run compute_train_rmse.py first!")
    elif not combined_path.exists():
        print(f"✗ Combined metrics file not found: {combined_path}")
    else:
        try:
            best_models = select_best_by_rmse_f1(rmse_path, combined_path)
            
            # Save per-channel results
            output_path = save_dir / f'best_model_by_rmse_f1_{args.metric_type}.csv'
            best_models.to_csv(output_path, index=False)
            
            print(f"✓ Saved: {output_path}")
            
            # Compute macro and micro averages
            required_cols = ['precision', 'recall', 'F1', 'F0.5', 'tp', 'fp', 'fn']
            if all(col in best_models.columns for col in required_cols):
                best_models_clean = best_models.dropna(subset=required_cols)
                
                if len(best_models_clean) > 0:
                    stats = compute_macro_micro(best_models_clean)
                    
                    # Create summary dataframe
                    summary = pd.DataFrame([{
                        'selection_method': 'best_by_rmse_f1',
                        'metric_type': args.metric_type,
                        'num_channels': stats['num_channels'],
                        'macro_precision': stats['macro_precision'],
                        'macro_recall': stats['macro_recall'],
                        'macro_F1': stats['macro_F1'],
                        'macro_F0.5': stats['macro_F0.5'],
                        'micro_precision': stats['micro_precision'],
                        'micro_recall': stats['micro_recall'],
                        'micro_F1': stats['micro_F1'],
                        'micro_F0.5': stats['micro_F0.5'],
                        'total_tp': stats['total_tp'],
                        'total_fp': stats['total_fp'],
                        'total_fn': stats['total_fn'],
                    }])
                    
                    # Save summary
                    summary_path = save_dir / f'best_model_summary_{args.metric_type}.csv'
                    summary.to_csv(summary_path, index=False)
                    
                    print(f"✓ Saved summary: {summary_path}")
                    
                    print("\n--- Best Model Statistics ---")
                    print(f"Selection method: Lowest RMSE, then highest F1")
                    print(f"Metric type: {args.metric_type}")
                    print(f"Number of channels: {stats['num_channels']}")
                    print("\nMacro averages (mean of per-channel metrics):")
                    print(f"  Precision: {stats['macro_precision']:.4f}")
                    print(f"  Recall:    {stats['macro_recall']:.4f}")
                    print(f"  F1:        {stats['macro_F1']:.4f}")
                    print(f"  F0.5:      {stats['macro_F0.5']:.4f}")
                    print("\nMicro averages (aggregated TP/FP/FN):")
                    print(f"  Precision: {stats['micro_precision']:.4f}")
                    print(f"  Recall:    {stats['micro_recall']:.4f}")
                    print(f"  F1:        {stats['micro_F1']:.4f}")
                    print(f"  F0.5:      {stats['micro_F0.5']:.4f}")
                    print(f"\nTotal counts:")
                    print(f"  TP: {stats['total_tp']}")
                    print(f"  FP: {stats['total_fp']}")
                    print(f"  FN: {stats['total_fn']}")
                    
                    # Show per-channel breakdown
                    print("\n--- Best Model per Channel ---")
                    for _, row in best_models_clean.head(10).iterrows():
                        print(f"{row['dataset']}: {row['model']:<25} "
                              f"RMSE={row['rmse_train']:.6f}, F1={row['F1']:.4f}")
                    
                    if len(best_models_clean) > 10:
                        print(f"... and {len(best_models_clean) - 10} more channels")
                
            else:
                print("✗ Warning: Missing required columns for macro/micro computation")
                
        except Exception as e:
            print(f"✗ Error selecting best models by RMSE+F1: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"\nAll results saved to: {save_dir}")


if __name__ == "__main__":
    main()
