#!/usr/bin/env python
"""
Compute training RMSE for all models including OCSVM reconstruction.
OCSVM reconstruction uses (Kernel)PCA:
- RBF & POLY: KernelPCA with fit_inverse_transform=True
- LINEAR: Standard PCA
- Weighted ensemble: combines all three reconstructions
"""
import os
import argparse
import time
import datetime as dt
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA, PCA

from config import get_config_dict
from trainers import (
    train_predictor,
    infer_predictor,
    train_autoencoder,
    infer_autoencoder,
    train_vae,
    infer_vae
)
import torch


def parse_rmse_args():
    """Parse RMSE computation arguments."""
    parser = argparse.ArgumentParser(
        description='Compute Training RMSE for All Models'
    )
    
    # Paths
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to input CSV file')
    parser.add_argument('--output_dir', type=str, default='anom_exp_latest',
                        help='Output directory for results')
    
    # Data columns
    parser.add_argument('--features_prefix', type=str, default='feature_',
                        help='Prefix for feature columns')
    parser.add_argument('--chan_col', type=str, default='chan_id',
                        help='Channel ID column name')
    parser.add_argument('--subset_col', type=str, default='subset',
                        help='Subset column name (train/test)')
    parser.add_argument('--time_col', type=str, default='time_step',
                        help='Time step column name')
    
    # Model parameters (same as training)
    parser.add_argument('--window', type=int, default=120,
                        help='Window size for LSTM predictor')
    parser.add_argument('--ae_window', type=int, default=100,
                        help='Window size for AE/VAE')
    parser.add_argument('--hidden', type=int, default=128,
                        help='Hidden dimension size')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--z_dim', type=int, default=64,
                        help='Latent dimension for VAE')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers')
    
    # OCSVM PCA parameters
    parser.add_argument('--pca_variance_ratio', type=float, default=0.90,
                        help='Variance ratio for PCA components (e.g., 0.90 = 90%%)')
    
    # Ensemble weights for OCSVM
    parser.add_argument('--ocsvm_weight_rbf', type=float, default=0.31,
                        help='Weight for RBF reconstruction')
    parser.add_argument('--ocsvm_weight_linear', type=float, default=0.36,
                        help='Weight for Linear reconstruction')
    parser.add_argument('--ocsvm_weight_poly', type=float, default=0.33,
                        help='Weight for Poly reconstruction')
    
    # Runtime
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use')
    
    # Option to reuse saved reconstructions
    parser.add_argument('--reuse_saved', action='store_true',
                        help='Reuse saved reconstruction .npy files if available')
    
    return parser.parse_args()


def _rmse_train(X_true, X_pred, min_rmse=0.0100001):
    """Compute training RMSE."""
    X_true = np.asarray(X_true, np.float32)
    X_pred = np.asarray(X_pred, np.float32)
    T = min(len(X_true), len(X_pred))
    
    if T == 0:
        return float("nan")
    
    val = float(np.sqrt(np.mean((X_true[:T] - X_pred[:T])**2)))
    return max(min_rmse, val)


def _load_saved_recon(output_dir, prefix, cid):
    """Try to load saved reconstruction."""
    p = output_dir / f"{prefix}_train_recon_{cid}.npy"
    if p.exists():
        try:
            return np.load(p)
        except Exception:
            return None
    return None


def main():
    args = parse_rmse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f'Device: {device}')
    
    # Setup output directory
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
    
    # Build config dict for trainers
    config = {
        "window": args.window,
        "ae_window": args.ae_window,
        "hidden": args.hidden,
        "n_layers": args.n_layers,
        "dropout": args.dropout,
        "z_dim": args.z_dim,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "lr": args.lr,
        "num_workers": args.num_workers,
    }
    
    # Get channels
    channels = sorted(df[args.chan_col].unique().tolist())
    
    records = []
    t0 = time.time()
    
    print("\n=== Computing Training RMSE for All Models ===\n")
    
    for cid in tqdm(channels, desc="train_rmse_all_models"):
        gi = df.loc[df[args.chan_col] == cid].sort_values(args.time_col)
        tri = gi.loc[gi[args.subset_col] == "train"]
        
        if tri.empty:
            continue
        
        Xtr = tri[feature_cols].to_numpy(np.float32)
        d = Xtr.shape[1]
        
        # === 1. LSTM Predictor (M2AD backbone) ===
        try:
            if args.reuse_saved:
                Ytr = _load_saved_recon(output_dir, "lstm_pred", cid)
            else:
                Ytr = None
            
            if Ytr is None:
                from models import LSTMPredictor
                mdl = train_predictor(Xtr, args.window, d, config, device)
                Ytr = infer_predictor(mdl, Xtr, args.window, config, device)
                np.save(output_dir / f"lstm_pred_train_recon_{cid}.npy", Ytr)
            
            rmse_pred = _rmse_train(Xtr, Ytr)
            records.append({
                "channel": str(cid),
                "model": "LSTM+M2AD",
                "rmse_train": rmse_pred
            })
        except Exception as e:
            records.append({
                "channel": str(cid),
                "model": "LSTM+M2AD",
                "rmse_train": float("nan"),
                "note": str(e)
            })
        
        # === 2. LSTM-AE ===
        try:
            if args.reuse_saved:
                Rtr = _load_saved_recon(output_dir, "ae", cid)
            else:
                Rtr = None
            
            if Rtr is None:
                ae = train_autoencoder(Xtr, args.ae_window, d, config, device)
                Rtr = infer_autoencoder(ae, Xtr, args.ae_window, config, device)
                np.save(output_dir / f"ae_train_recon_{cid}.npy", Rtr)
            
            rmse_ae = _rmse_train(Xtr, Rtr)
            records.append({
                "channel": str(cid),
                "model": "LSTM-AE+M2AD",
                "rmse_train": rmse_ae
            })
            records.append({
                "channel": str(cid),
                "model": "LSTM-AE(standalone)",
                "rmse_train": rmse_ae
            })
        except Exception as e:
            for tag in ["LSTM-AE+M2AD", "LSTM-AE(standalone)"]:
                records.append({
                    "channel": str(cid),
                    "model": tag,
                    "rmse_train": float("nan"),
                    "note": str(e)
                })
        
        # === 3. LSTM-VAE ===
        try:
            if args.reuse_saved:
                VRtr = _load_saved_recon(output_dir, "vae", cid)
            else:
                VRtr = None
            
            if VRtr is None:
                vae = train_vae(Xtr, args.ae_window, d, config, device)
                VRtr = infer_vae(vae, Xtr, args.ae_window, config, device)
                np.save(output_dir / f"vae_train_recon_{cid}.npy", VRtr)
            
            rmse_vae = _rmse_train(Xtr, VRtr)
            records.append({
                "channel": str(cid),
                "model": "LSTM-VAE+M2AD",
                "rmse_train": rmse_vae
            })
            records.append({
                "channel": str(cid),
                "model": "LSTM-VAE(standalone)",
                "rmse_train": rmse_vae
            })
        except Exception as e:
            for tag in ["LSTM-VAE+M2AD", "LSTM-VAE(standalone)"]:
                records.append({
                    "channel": str(cid),
                    "model": tag,
                    "rmse_train": float("nan"),
                    "note": str(e)
                })
        
        # === 4. OCSVM Reconstructions via (Kernel)PCA ===
        try:
            # Scale data
            ss = StandardScaler().fit(Xtr)
            Xtr_s = ss.transform(Xtr)
            
            # Determine n_components (90% of features)
            n_comp = max(1, min(d, int(np.ceil(args.pca_variance_ratio * d))))
            
            # 4a. RBF KernelPCA
            kpca_rbf = KernelPCA(
                n_components=n_comp,
                kernel="rbf",
                gamma=None,
                fit_inverse_transform=True,
                eigen_solver="auto",
                random_state=args.seed
            )
            Z_rbf = kpca_rbf.fit_transform(Xtr_s)
            Xr_rbf_s = kpca_rbf.inverse_transform(Z_rbf)
            Xr_rbf = ss.inverse_transform(Xr_rbf_s)
            
            rmse_rbf = _rmse_train(Xtr, Xr_rbf)
            records.append({
                "channel": str(cid),
                "model": "OCSVM_rbf(KPCA)",
                "rmse_train": rmse_rbf
            })
            np.save(output_dir / f"ocsvm_kpca_rbf_train_recon_{cid}.npy", Xr_rbf)
            
            # 4b. Poly KernelPCA
            kpca_poly = KernelPCA(
                n_components=n_comp,
                kernel="poly",
                degree=2,
                gamma=None,
                coef0=1.0,
                fit_inverse_transform=True,
                eigen_solver="auto",
                random_state=args.seed
            )
            Z_poly = kpca_poly.fit_transform(Xtr_s)
            Xr_poly_s = kpca_poly.inverse_transform(Z_poly)
            Xr_poly = ss.inverse_transform(Xr_poly_s)
            
            rmse_poly = _rmse_train(Xtr, Xr_poly)
            records.append({
                "channel": str(cid),
                "model": "OCSVM_poly(KPCA)",
                "rmse_train": rmse_poly
            })
            np.save(output_dir / f"ocsvm_kpca_poly_train_recon_{cid}.npy", Xr_poly)
            
            # 4c. Linear PCA
            pca_lin = PCA(n_components=min(d, n_comp), random_state=args.seed)
            Z_lin = pca_lin.fit_transform(Xtr_s)
            Xr_lin_s = pca_lin.inverse_transform(Z_lin)
            Xr_lin = ss.inverse_transform(Xr_lin_s)
            
            rmse_lin = _rmse_train(Xtr, Xr_lin)
            records.append({
                "channel": str(cid),
                "model": "OCSVM_linear(PCA)",
                "rmse_train": rmse_lin
            })
            np.save(output_dir / f"ocsvm_pca_linear_train_recon_{cid}.npy", Xr_lin)
            
            # 4d. Weighted ensemble reconstruction
            w_rbf = args.ocsvm_weight_rbf
            w_lin = args.ocsvm_weight_linear
            w_poly = args.ocsvm_weight_poly
            
            Xr_soft = w_rbf * Xr_rbf + w_lin * Xr_lin + w_poly * Xr_poly
            rmse_soft = _rmse_train(Xtr, Xr_soft)
            records.append({
                "channel": str(cid),
                "model": "OCSVMSOFT_w(KPCA/PCA)",
                "rmse_train": rmse_soft
            })
            np.save(output_dir / f"ocsvm_soft_train_recon_{cid}.npy", Xr_soft)
            
        except Exception as e:
            for tag in ["OCSVM_rbf(KPCA)", "OCSVM_poly(KPCA)", 
                       "OCSVM_linear(PCA)", "OCSVMSOFT_w(KPCA/PCA)"]:
                records.append({
                    "channel": str(cid),
                    "model": tag,
                    "rmse_train": float("nan"),
                    "note": f"recon failed: {e}"
                })
    
    # Save results
    rmse_df = pd.DataFrame(records)
    
    # Per-channel results
    def _safe_write_csv(df, filename):
        """Write CSV with timestamp fallback if permission denied."""
        p0 = output_dir / filename
        try:
            df.to_csv(p0, index=False)
            return p0
        except PermissionError:
            ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            base, ext = os.path.splitext(filename)
            p1 = output_dir / f"{base}_{ts}{ext}"
            df.to_csv(p1, index=False)
            return p1
    
    rmse_per = _safe_write_csv(rmse_df, "train_rmse_per_channel.csv")
    
    # Summary by model
    summary = (
        rmse_df.dropna(subset=["rmse_train"])
        .groupby("model", as_index=False)
        .agg(
            mean_rmse=("rmse_train", "mean"),
            median_rmse=("rmse_train", "median"),
            channels=("channel", "nunique")
        )
        .sort_values("mean_rmse")
    )
    
    rmse_sum = _safe_write_csv(summary, "train_rmse_summary.csv")
    
    elapsed = time.time() - t0
    print(f"\n[TRAIN RMSE] wrote per-channel → {rmse_per}")
    print(f"[TRAIN RMSE] wrote summary     → {rmse_sum}")
    print(f"Completed {len(channels)} channels in {elapsed:.1f}s")
    
    print("\n=== Summary ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
