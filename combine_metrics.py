#!/usr/bin/env python
"""
Combine all metrics (EVENT and EXACT) from all models.
Compute overall scores per model by aggregating TP/FP/FN across channels.
Replicates Cell A from the notebook.
"""
import os
import glob
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Combine metrics from all models and compute overall scores'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='anom_exp_latest',
        help='Output directory containing results (default: anom_exp_latest)'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default=None,
        help='Directory to save combined results (default: same as output_dir)'
    )
    return parser.parse_args()


def prf_from_counts(tp, fp, fn, beta=1.0):
    """Compute precision, recall, F-beta from aggregated counts."""
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


def combine_and_aggregate(output_dir, metric_type='event'):
    """
    Combine all metric files and compute overall scores.
    
    Args:
        output_dir: Path to results directory
        metric_type: 'event' or 'exact'
    
    Returns:
        combined_df: All metrics combined
        overall_df: Overall scores per model
    """
    # Find all metric files
    if metric_type == 'event':
        pattern = os.path.join(output_dir, '**/*event_metrics.csv')
    else:
        pattern = os.path.join(output_dir, '**/*exact_metrics.csv')
    
    metric_paths = glob.glob(pattern, recursive=True)
    
    if not metric_paths:
        raise RuntimeError(
            f"No {metric_type}_metrics.csv files found in {output_dir}"
        )
    
    print(f"\nFound {len(metric_paths)} {metric_type} metrics files:")
    for p in metric_paths:
        print(f"  - {os.path.relpath(p, output_dir)}")
    
    # Load and combine all metrics
    dfs = []
    for p in metric_paths:
        try:
            df = pd.read_csv(p)
            df['source_file'] = os.path.relpath(p, output_dir)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not read {p}: {e}")
    
    if not dfs:
        raise RuntimeError("No valid metric files could be read")
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Ensure required columns exist
    required_cols = {'model', 'dataset', 'tp', 'fp', 'fn'}
    missing = required_cols - set(combined_df.columns)
    if missing:
        raise AssertionError(
            f"Missing required columns in metrics: {missing}"
        )
    
    # Ensure numeric columns are numeric
    for col in ['tp', 'fp', 'fn', 'precision', 'recall', 'F1', 'F0.5']:
        if col in combined_df.columns:
            combined_df[col] = pd.to_numeric(
                combined_df[col],
                errors='coerce'
            )
    
    # Aggregate per model across all channels
    grouped = combined_df.groupby('model', as_index=False)[
        ['tp', 'fp', 'fn']
    ].sum()
    
    # Recompute metrics from aggregated counts
    overall_rows = []
    for _, row in grouped.iterrows():
        model = row['model']
        tp = int(row['tp'])
        fp = int(row['fp'])
        fn = int(row['fn'])
        
        # Compute F1 and F0.5
        p, r, f1 = prf_from_counts(tp, fp, fn, beta=1.0)
        p05, r05, f05 = prf_from_counts(tp, fp, fn, beta=0.5)
        
        overall_rows.append({
            'model': model,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': p,
            'recall': r,
            'F1': f1,
            'F0.5': f05
        })
    
    overall_df = pd.DataFrame(overall_rows)
    
    # Sort by F0.5 descending
    overall_df = overall_df.sort_values('F0.5', ascending=False)
    
    return combined_df, overall_df


def main():
    args = parse_args()
    
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")
    
    # Determine save directory
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = output_dir
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("COMBINING AND AGGREGATING METRICS")
    print("=" * 70)
    
    # ========== EVENT METRICS ==========
    print("\n[1/2] Processing EVENT metrics...")
    try:
        event_combined, event_overall = combine_and_aggregate(
            output_dir,
            metric_type='event'
        )
        
        # Save event results
        event_combined_path = save_dir / 'all_event_metrics_combined.csv'
        event_overall_path = save_dir / 'overall_event_metrics_by_model.csv'
        
        event_combined.to_csv(event_combined_path, index=False)
        event_overall.to_csv(event_overall_path, index=False)
        
        print(f"\n✓ Event metrics combined: {event_combined_path}")
        print(f"✓ Event overall scores:   {event_overall_path}")
        
        print("\n--- EVENT: Overall Scores by Model (sorted by F0.5) ---")
        print(event_overall.to_string(index=False))
        
    except Exception as e:
        print(f"\n✗ Error processing event metrics: {e}")
    
    # ========== EXACT METRICS ==========
    print("\n" + "=" * 70)
    print("\n[2/2] Processing EXACT metrics...")
    try:
        exact_combined, exact_overall = combine_and_aggregate(
            output_dir,
            metric_type='exact'
        )
        
        # Save exact results
        exact_combined_path = save_dir / 'all_exact_metrics_combined.csv'
        exact_overall_path = save_dir / 'overall_exact_metrics_by_model.csv'
        
        exact_combined.to_csv(exact_combined_path, index=False)
        exact_overall.to_csv(exact_overall_path, index=False)
        
        print(f"\n✓ Exact metrics combined: {exact_combined_path}")
        print(f"✓ Exact overall scores:   {exact_overall_path}")
        
        print("\n--- EXACT: Overall Scores by Model (sorted by F0.5) ---")
        print(exact_overall.to_string(index=False))
        
    except Exception as e:
        print(f"\n✗ Error processing exact metrics: {e}")
    
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"\nAll results saved to: {save_dir}")


if __name__ == "__main__":
    main()
