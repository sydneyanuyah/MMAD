#!/usr/bin/env python
"""
Train standalone LSTM Autoencoder (without M2AD).
Uses simple MSE threshold for anomaly detection.
"""
import os
import json
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import parse_args, setup_environment, get_config_dict
from data_loader import load_and_prepare_data
from trainers import train_autoencoder, infer_autoencoder
from utils import (
    canonical_postprocess_labels,
    event_scores,
    prf
)


def _ewma_1d(x, alpha):
    """Apply EWMA to 1D array."""
    if len(x) == 0:
        return np.array([], dtype=np.float32)
    out = np.empty_like(x, dtype=np.float32)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i-1]
    return out


def _pointwise_mse(X, R):
    """Compute point-wise MSE."""
    return ((X - R) ** 2).mean(axis=1).astype(np.float32)


def _threshold_from_train(err_tr, fpr):
    """Compute threshold from training errors."""
    q = 1.0 - float(fpr)
    q = min(max(q, 0.5), 0.999999)
    return float(np.quantile(err_tr, q))


def main():
    # Parse arguments and setup
    args = parse_args()
    device, output_dir = setup_environment(args)
    config = get_config_dict(args)
    
    # Load data
    channels, df = load_and_prepare_data(args.data_path, config)
    
    # Training parameters
    w = config["ae_window"]
    hidden = config["hidden"]
    n_layers = config["n_layers"]
    dropout = config["dropout"]
    ewma_alpha = config["ewma_alpha"]
    target_fpr = config["target_fpr"]
    
    print("\nTraining Standalone LSTM Autoencoder")
    print(f"AE Window: {w}, Hidden: {hidden}, Layers: {n_layers}")
    print(f"EWMA alpha: {ewma_alpha}, Target FPR: {target_fpr}")
    
    # Storage for results
    rows_exact = []
    rows_event = []
    thresholds = {}
    
    t0 = time.time()
    
    for chan_id in tqdm(sorted(channels.keys()), desc="lstm_ae_standalone"):
        data = channels[chan_id]
        Xtr = data["X_train"]
        Xte = data["X_test"]
        y_true = data["y_test"]
        
        d = Xtr.shape[1]
        
        # Train autoencoder
        try:
            model = train_autoencoder(Xtr, w, d, config, device)
            if model is None:
                continue
            
            # Inference (reconstruction)
            Rtr = infer_autoencoder(model, Xtr, w, config, device)
            Rte = infer_autoencoder(model, Xte, w, config, device)
            
            # Compute errors
            e_tr = _pointwise_mse(Xtr, Rtr)
            e_te = _pointwise_mse(Xte, Rte)
            
            # Smooth test errors
            e_te_s = _ewma_1d(e_te, ewma_alpha)
            
            # Compute threshold from training
            thr = _threshold_from_train(e_tr, target_fpr)
            
            # Predict
            yhat_full = (e_te_s >= thr).astype(int)
            
            # Post-process predictions
            yhat_full = canonical_postprocess_labels(
                yhat_full,
                dilate_radius=config["dilate_radius"],
                min_event_len=config["min_event_len"]
            )
            
            # Align for metrics
            L = min(len(y_true), len(yhat_full))
            y_true_m = y_true[:L]
            yhat_m = yhat_full[:L]
            
            # Point-wise metrics
            tp = int(((yhat_m == 1) & (y_true_m == 1)).sum())
            fp = int(((yhat_m == 1) & (y_true_m == 0)).sum())
            fn = int(((yhat_m == 0) & (y_true_m == 1)).sum())
            
            p, r, f1 = prf(tp, fp, fn, beta=1.0)
            f05 = prf(tp, fp, fn, beta=0.5)[2]
            
            rows_exact.append({
                "model": "LSTM-AE(standalone)",
                "dataset": chan_id,
                "precision": p,
                "recall": r,
                "F1": f1,
                "F0.5": f05,
                "tp": tp,
                "fp": fp,
                "fn": fn
            })
            
            # Event-wise metrics
            tp_e, fp_e, fn_e = event_scores(y_true_m, yhat_m)
            pe, re, f1e = prf(tp_e, fp_e, fn_e, beta=1.0)
            f05e = prf(tp_e, fp_e, fn_e, beta=0.5)[2]
            
            rows_event.append({
                "model": "LSTM-AE(standalone-ev)",
                "dataset": chan_id,
                "precision": pe,
                "recall": re,
                "F1": f1e,
                "F0.5": f05e,
                "tp": tp_e,
                "fp": fp_e,
                "fn": fn_e
            })
            
            # Save threshold
            thresholds[chan_id] = {
                "threshold": float(thr),
                "ewma_alpha": float(ewma_alpha),
                "target_fpr": float(target_fpr),
                "ae_window": int(w)
            }
            
            # Save reconstruction and errors (optional)
            np.save(output_dir / f"ae_train_recon_{chan_id}.npy", Rtr)
            np.save(output_dir / f"ae_test_recon_{chan_id}.npy", Rte)
            np.save(output_dir / f"ae_train_err_{chan_id}.npy", e_tr)
            np.save(output_dir / f"ae_test_err_{chan_id}.npy", e_te)
            
        except Exception as e:
            print(f"[AE] skip {chan_id}: {e}")
            continue
    
    # Save results
    exact_df = pd.DataFrame(rows_exact)
    event_df = pd.DataFrame(rows_event)
    
    exact_df.to_csv(output_dir / "ae_standalone_exact_metrics.csv", index=False)
    event_df.to_csv(output_dir / "ae_standalone_event_metrics.csv", index=False)
    
    with open(output_dir / "ae_standalone_thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    
    elapsed = time.time() - t0
    print(f"\nCompleted {len(channels)} channels in {elapsed:.1f}s")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
