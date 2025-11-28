#!/usr/bin/env python
"""
Train LSTM VAE + M2AD anomaly detector.
"""
import os
import json
import time
import pandas as pd
from tqdm import tqdm

from config import parse_args, setup_environment, get_config_dict
from data_loader import load_and_prepare_data
from trainers import train_vae, infer_vae
from m2ad_scorer import M2ADScorer
from utils import (
    canonical_postprocess_labels,
    event_scores,
    prf
)


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
    z_dim = config["z_dim"]
    n_layers = config["n_layers"]
    dropout = config["dropout"]
    
    # M2AD parameters
    error_mode = config["error_mode"]
    area_l = config["area_l"]
    ewma_alpha = config["ewma_alpha"]
    kmax = config["gmm_components_max"]
    target_fpr = config["target_fpr"]
    min_p = config["min_p"]
    
    print("\nTraining LSTM VAE + M2AD")
    print(f"VAE Window: {w}, Hidden: {hidden}, Z-dim: {z_dim}, Layers: {n_layers}")
    print(f"Error mode: {error_mode}, EWMA alpha: {ewma_alpha}")
    
    # Storage for results
    rows_exact = []
    rows_event = []
    fit_dump = {}
    
    t0 = time.time()
    
    for chan_id in tqdm(sorted(channels.keys()), desc="lstm_vae_m2ad"):
        data = channels[chan_id]
        Xtr = data["X_train"]
        Xte = data["X_test"]
        y_true = data["y_test"]
        
        d = Xtr.shape[1]
        
        # Train VAE
        model = train_vae(Xtr, w, d, config, device)
        if model is None:
            continue
        
        # Inference (reconstruction)
        Rtr = infer_vae(model, Xtr, w, config, device)
        Rte = infer_vae(model, Xte, w, config, device)
        
        # Align lengths
        Ttr = min(len(Xtr), len(Rtr))
        Xtr_a, Rtr_a = Xtr[:Ttr], Rtr[:Ttr]
        
        Tte = min(len(Xte), len(Rte), len(y_true))
        Xte_a, Rte_a = Xte[:Tte], Rte[:Tte]
        y_true_a = y_true[:Tte]
        
        # Fit M2AD scorer
        scorer = M2ADScorer(
            error_mode=error_mode,
            area_l=area_l,
            ewma_alpha=ewma_alpha,
            kmax=kmax,
            target_fpr=target_fpr,
            min_p=min_p,
            weights=None,
            seed=args.seed
        ).fit(Xtr_a, Rtr_a)
        
        # Score test data
        out = scorer.score(Xte_a, Rte_a)
        
        # Post-process predictions
        yhat_full = out["label"].astype(int)
        yhat_full = canonical_postprocess_labels(
            yhat_full,
            dilate_radius=config["dilate_radius"],
            min_event_len=config["min_event_len"]
        )
        
        # Align for metrics
        L = min(len(y_true_a), len(yhat_full))
        y_true_m = y_true_a[:L]
        yhat_m = yhat_full[:L]
        
        # Point-wise metrics
        tp = int(((yhat_m == 1) & (y_true_m == 1)).sum())
        fp = int(((yhat_m == 1) & (y_true_m == 0)).sum())
        fn = int(((yhat_m == 0) & (y_true_m == 1)).sum())
        
        p, r, f1 = prf(tp, fp, fn, beta=1.0)
        f05 = prf(tp, fp, fn, beta=0.5)[2]
        
        rows_exact.append({
            "model": "LSTM-VAE+M2AD",
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
            "model": "LSTM-VAE+M2AD(ev)",
            "dataset": chan_id,
            "precision": pe,
            "recall": re,
            "F1": f1e,
            "F0.5": f05e,
            "tp": tp_e,
            "fp": fp_e,
            "fn": fn_e
        })
        
        # Save fit parameters
        alpha, theta = scorer.gamma
        fit_dump[chan_id] = {
            "threshold": float(out["threshold"]),
            "gamma_alpha": float(alpha),
            "gamma_theta": float(theta),
            "vae_window": int(w),
            "z_dim": int(z_dim),
            "error_mode": scorer.error_mode,
            "area_l": int(scorer.area_l),
            "ewma_alpha": float(scorer.ewma_alpha),
            "gmm_components_used": int(len(scorer.gmms)),
        }
    
    # Save results
    exact_df = pd.DataFrame(rows_exact)
    event_df = pd.DataFrame(rows_event)
    
    exact_df.to_csv(output_dir / "vae_m2ad_exact_metrics.csv", index=False)
    event_df.to_csv(output_dir / "vae_m2ad_event_metrics.csv", index=False)
    
    with open(output_dir / "vae_m2ad_fit_params.json", "w") as f:
        json.dump(fit_dump, f, indent=2)
    
    elapsed = time.time() - t0
    print(f"\nCompleted {len(channels)} channels in {elapsed:.1f}s")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
