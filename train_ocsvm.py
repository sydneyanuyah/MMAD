#!/usr/bin/env python
"""
Train One-Class SVM anomaly detectors with multiple kernels.
Includes: RBF, Linear, Poly kernels and 3-model ensemble.
"""
import os
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

from utils import (
    ranges_from_labels,
    dilate_binary,
    remove_short_events,
    event_scores,
    prf
)


def parse_ocsvm_args():
    """Parse OCSVM-specific arguments."""
    parser = argparse.ArgumentParser(description='One-Class SVM Anomaly Detection')
    
    # Paths
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to input CSV file')
    parser.add_argument('--output_dir', type=str, default='anom_exp_latest',
                        help='Output directory for results')
    
    # Data columns
    parser.add_argument('--features_prefix', type=str, default='feature_',
                        help='Prefix for feature columns')
    parser.add_argument('--label_col', type=str, default='anomaly_label',
                        help='Label column name')
    parser.add_argument('--chan_col', type=str, default='chan_id',
                        help='Channel ID column name')
    parser.add_argument('--subset_col', type=str, default='subset',
                        help='Subset column name (train/test)')
    parser.add_argument('--time_col', type=str, default='time_step',
                        help='Time step column name')
    
    # OCSVM parameters - RBF
    parser.add_argument('--rbf_nu', type=float, default=0.005,
                        help='Nu parameter for RBF kernel')
    parser.add_argument('--rbf_gamma', type=str, default='auto',
                        help='Gamma parameter for RBF kernel')
    
    # OCSVM parameters - Linear
    parser.add_argument('--linear_nu', type=float, default=0.001,
                        help='Nu parameter for linear kernel')
    
    # OCSVM parameters - Poly
    parser.add_argument('--poly_nu', type=float, default=0.005,
                        help='Nu parameter for poly kernel')
    parser.add_argument('--poly_gamma', type=str, default='auto',
                        help='Gamma parameter for poly kernel')
    parser.add_argument('--poly_degree', type=int, default=2,
                        help='Degree for poly kernel')
    parser.add_argument('--poly_coef0', type=float, default=1.0,
                        help='Coef0 parameter for poly kernel')
    
    # Ensemble weights (should sum to 1.0)
    parser.add_argument('--ensemble_weight_rbf', type=float, default=0.31,
                        help='Weight for RBF in ensemble')
    parser.add_argument('--ensemble_weight_linear', type=float, default=0.36,
                        help='Weight for linear in ensemble')
    parser.add_argument('--ensemble_weight_poly', type=float, default=0.33,
                        help='Weight for poly in ensemble')
    
    # Training
    parser.add_argument('--max_train_samples', type=int, default=250000,
                        help='Max training samples to use')
    
    # Event post-processing (OCSVM uses min_len=50)
    parser.add_argument('--min_event_len', type=int, default=50,
                        help='Minimum event length')
    parser.add_argument('--dilate_radius', type=int, default=1,
                        help='Dilation radius for events')
    parser.add_argument('--merge_gap', type=int, default=3,
                        help='Minimum gap to merge events')
    
    # Runtime
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def postprocess_events(y, min_len=50, pad=1, min_gap=3):
    """
    Canonical prediction postprocessing for OCSVM:
    1) dilate
    2) merge close segments
    3) drop very short events
    """
    y = np.asarray(y).astype(int)
    y = dilate_binary(y, radius=pad)
    
    # Merge close events
    evs = ranges_from_labels(y)
    if len(evs) > 1:
        for (s1, e1), (s2, e2) in zip(evs[:-1], evs[1:]):
            gap = s2 - e1 - 1
            if 0 < gap <= min_gap:
                y[e1+1:s2] = 1
    
    y = remove_short_events(y, min_len=min_len)
    return y


def _align_for_metrics(y_true, y_pred):
    """Align labels for metrics computation."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    L = min(len(y_true), len(y_pred))
    return y_true[:L], y_pred[:L]


def _add_invariants(row_dict, y_true, y_pred, event_mode=False):
    """Add invariant counts to metrics row."""
    y_true, y_pred = _align_for_metrics(y_true, y_pred)
    row_dict["n_true"] = int(len(y_true))
    row_dict["n_pred"] = int(len(y_pred))
    
    if event_mode:
        n_events = len(ranges_from_labels(y_true))
        row_dict["n_pos"] = int(n_events)
        row_dict["tp_plus_fn"] = int(row_dict.get("tp", 0) + row_dict.get("fn", 0))
    else:
        n_pos = int((y_true == 1).sum())
        row_dict["n_pos"] = n_pos
        row_dict["tp_plus_fn"] = n_pos
    
    return row_dict


def run_ocsvm_kernel(df, feature_cols, kernel_name, kernel_params, 
                     config, output_dir):
    """Train and evaluate OCSVM with a given kernel."""
    channels = sorted(df[config["chan_col"]].unique().tolist())
    results = {}
    rows_exact, rows_event = [], []
    
    for cid in tqdm(channels, desc=f"ocsvm_{kernel_name}_metrics"):
        g = df.loc[df[config["chan_col"]] == cid].sort_values(config["time_col"])
        tr = g.loc[g[config["subset_col"]] == "train"]
        te = g.loc[g[config["subset_col"]] == "test"]
        
        if tr.empty or te.empty:
            continue
        
        Xtr = tr[feature_cols].to_numpy(np.float32)
        Xte = te[feature_cols].to_numpy(np.float32)
        y_true = te[config["label_col"]].to_numpy(np.int32)
        t_test = te[config["time_col"]].to_numpy()
        
        # Scale
        ss = StandardScaler().fit(Xtr)
        Xtr_s = ss.transform(Xtr)
        Xte_s = ss.transform(Xte)
        
        # Subsample if needed
        if len(Xtr_s) > config["max_train_samples"]:
            idx = np.linspace(0, len(Xtr_s)-1, config["max_train_samples"]).astype(int)
            Xtr_fit = Xtr_s[idx]
        else:
            Xtr_fit = Xtr_s
        
        # Build and fit classifier
        clf = OneClassSVM(**kernel_params)
        clf.fit(Xtr_fit)
        
        # Predict
        scores = clf.decision_function(Xte_s).ravel()
        y_pred_full = (scores < 0.0).astype(np.int32)
        
        # Post-process
        y_pred_full = postprocess_events(
            y_pred_full,
            min_len=config["min_event_len"],
            pad=config["dilate_radius"],
            min_gap=config["merge_gap"]
        )
        
        # Exact metrics
        y_true_m, y_pred_m = _align_for_metrics(y_true, y_pred_full)
        tp = int(((y_pred_m == 1) & (y_true_m == 1)).sum())
        fp = int(((y_pred_m == 1) & (y_true_m == 0)).sum())
        fn = int(((y_pred_m == 0) & (y_true_m == 1)).sum())
        
        p, r, f1 = prf(tp, fp, fn, beta=1.0)
        f05 = prf(tp, fp, fn, beta=0.5)[2]
        
        rows_exact.append(_add_invariants({
            "model": f"OCSVM_{kernel_name}",
            "dataset": str(cid),
            "precision": p,
            "recall": r,
            "F1": f1,
            "F0.5": f05,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }, y_true_m, y_pred_m, event_mode=False))
        
        # Event metrics
        tp_e, fp_e, fn_e = event_scores(y_true_m, y_pred_m)
        pe, re, f1e = prf(tp_e, fp_e, fn_e, beta=1.0)
        f05e = prf(tp_e, fp_e, fn_e, beta=0.5)[2]
        
        rows_event.append(_add_invariants({
            "model": f"OCSVM_{kernel_name}(ev)",
            "dataset": str(cid),
            "precision": pe,
            "recall": re,
            "F1": f1e,
            "F0.5": f05e,
            "tp": tp_e,
            "fp": fp_e,
            "fn": fn_e,
        }, y_true_m, y_pred_m, event_mode=True))
        
        # Save per-channel scores
        dump = pd.DataFrame({
            "time_step": t_test,
            "score": scores,
            "y_pred": y_pred_full,
            "y_true": y_true,
        })
        dump.to_csv(
            output_dir / f"ocsvm_{kernel_name}_{cid}_scores.csv",
            index=False
        )
        
        results[cid] = {
            "time": t_test,
            "scores": scores,
            "y_pred": y_pred_full,
            "y_true": y_true,
        }
    
    # Save metrics
    exact_df = pd.DataFrame(rows_exact)
    event_df = pd.DataFrame(rows_event)
    
    exact_out = output_dir / f"ocsvm_{kernel_name}_metrics_exact.csv"
    event_out = output_dir / f"ocsvm_{kernel_name}_metrics_event.csv"
    
    exact_df.to_csv(exact_out, index=False)
    event_df.to_csv(event_out, index=False)
    
    print(f"[done] OCSVM-{kernel_name}: saved {exact_out} and {event_out}")
    
    return results


def main():
    args = parse_ocsvm_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data_path = Path(args.data_path)
    assert data_path.exists(), f"Cannot find {data_path}"
    
    df = pd.read_csv(data_path)
    
    # Get feature columns
    feature_cols = [c for c in df.columns if c.startswith(args.features_prefix)]
    assert feature_cols, "No feature_* columns found."
    
    print(f"Loaded {len(df):,} rows, {len(feature_cols)} features")
    
    # Config dict
    config = {
        "chan_col": args.chan_col,
        "subset_col": args.subset_col,
        "time_col": args.time_col,
        "label_col": args.label_col,
        "max_train_samples": args.max_train_samples,
        "min_event_len": args.min_event_len,
        "dilate_radius": args.dilate_radius,
        "merge_gap": args.merge_gap,
    }
    
    # Save config
    with open(output_dir / 'ocsvm_config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print("\n=== Running OCSVM with Multiple Kernels ===\n")
    
    # 1. RBF kernel
    print(f"[1/3] Training RBF kernel (nu={args.rbf_nu})")
    rbf_params = {
        "kernel": "rbf",
        "gamma": args.rbf_gamma,
        "nu": args.rbf_nu,
    }
    rbf_results = run_ocsvm_kernel(
        df, feature_cols, "rbf", rbf_params, config, output_dir
    )
    
    # 2. Linear kernel
    print(f"\n[2/3] Training Linear kernel (nu={args.linear_nu})")
    linear_params = {
        "kernel": "linear",
        "nu": args.linear_nu,
    }
    linear_results = run_ocsvm_kernel(
        df, feature_cols, "linear", linear_params, config, output_dir
    )
    
    # 3. Poly kernel
    print(f"\n[3/3] Training Poly kernel (nu={args.poly_nu}, degree={args.poly_degree})")
    poly_params = {
        "kernel": "poly",
        "degree": args.poly_degree,
        "gamma": args.poly_gamma,
        "coef0": args.poly_coef0,
        "nu": args.poly_nu,
    }
    poly_results = run_ocsvm_kernel(
        df, feature_cols, "poly", poly_params, config, output_dir
    )
    
    # 4. Ensemble (3-model majority vote)
    print("\n=== Building 3-Model Ensemble (Majority Vote) ===\n")
    
    rows_exact_ens, rows_event_ens = [], []
    common_cids = sorted(
        set(rbf_results.keys()) & 
        set(linear_results.keys()) & 
        set(poly_results.keys())
    )
    
    for cid in tqdm(common_cids, desc="ocsvm_ensemble3_MAJ_metrics"):
        A = rbf_results[cid]
        B = linear_results[cid]
        C = poly_results[cid]
        
        # Sanity check
        if not (np.array_equal(A["time"], B["time"]) and 
                np.array_equal(B["time"], C["time"])):
            print(f"Warning: time mismatch for {cid}, skipping")
            continue
        
        times = A["time"]
        y_true = A["y_true"]
        
        # Majority vote
        ya, yb, yc = A["y_pred"], B["y_pred"], C["y_pred"]
        y_maj_full = ((ya + yb + yc) >= 2).astype(np.int32)
        
        # Post-process
        y_maj_full = postprocess_events(
            y_maj_full,
            min_len=config["min_event_len"],
            pad=config["dilate_radius"],
            min_gap=config["merge_gap"]
        )
        
        # Exact metrics
        y_true_m, y_maj = _align_for_metrics(y_true, y_maj_full)
        tp = int(((y_maj == 1) & (y_true_m == 1)).sum())
        fp = int(((y_maj == 1) & (y_true_m == 0)).sum())
        fn = int(((y_maj == 0) & (y_true_m == 1)).sum())
        
        p, r, f1 = prf(tp, fp, fn, beta=1.0)
        f05 = prf(tp, fp, fn, beta=0.5)[2]
        
        rows_exact_ens.append(_add_invariants({
            "model": "OCSVM_ens3_MAJ",
            "dataset": str(cid),
            "precision": p,
            "recall": r,
            "F1": f1,
            "F0.5": f05,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }, y_true_m, y_maj, event_mode=False))
        
        # Event metrics
        tp_e, fp_e, fn_e = event_scores(y_true_m, y_maj)
        pe, re, f1e = prf(tp_e, fp_e, fn_e, beta=1.0)
        f05e = prf(tp_e, fp_e, fn_e, beta=0.5)[2]
        
        rows_event_ens.append(_add_invariants({
            "model": "OCSVM_ens3_MAJ(ev)",
            "dataset": str(cid),
            "precision": pe,
            "recall": re,
            "F1": f1e,
            "F0.5": f05e,
            "tp": tp_e,
            "fp": fp_e,
            "fn": fn_e,
        }, y_true_m, y_maj, event_mode=True))
    
    # Save ensemble metrics
    ens_exact_df = pd.DataFrame(rows_exact_ens)
    ens_event_df = pd.DataFrame(rows_event_ens)
    
    ens_exact_out = output_dir / "ocsvm_ens3_MAJ_metrics_exact.csv"
    ens_event_out = output_dir / "ocsvm_ens3_MAJ_metrics_event.csv"
    
    ens_exact_df.to_csv(ens_exact_out, index=False)
    ens_event_df.to_csv(ens_event_out, index=False)
    
    print(f"\n[done] OCSVM ensemble: saved {ens_exact_out} and {ens_event_out}")
    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
