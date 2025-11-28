"""
Configuration module for anomaly detection experiments.
"""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Anomaly Detection Training')
    
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
    
    # Windows
    parser.add_argument('--window', type=int, default=120,
                        help='Window size for LSTM predictor')
    parser.add_argument('--ae_window', type=int, default=100,
                        help='Window size for AE/VAE')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    
    # Model architecture
    parser.add_argument('--hidden', type=int, default=128,
                        help='Hidden dimension size')
    parser.add_argument('--n_layers', type=int, default=4,
                        help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--z_dim', type=int, default=64,
                        help='Latent dimension for VAE')
    
    # Discrepancy & smoothing
    parser.add_argument('--error_mode', type=str, default='area',
                        choices=['point', 'area'],
                        help='Error computation mode')
    parser.add_argument('--area_l', type=int, default=2,
                        help='Half-window for area integral')
    parser.add_argument('--ewma_alpha', type=float, default=0.3,
                        help='EWMA smoothing alpha')
    
    # M2AD calibration
    parser.add_argument('--gmm_components_max', type=int, default=3,
                        help='Max GMM components')
    parser.add_argument('--target_fpr', type=float, default=0.01,
                        help='Target false positive rate')
    parser.add_argument('--min_p', type=float, default=1e-12,
                        help='Minimum p-value')
    
    # Event post-processing
    parser.add_argument('--min_event_len', type=int, default=5,
                        help='Minimum event length')
    parser.add_argument('--dilate_radius', type=int, default=1,
                        help='Dilation radius for events')
    
    # Runtime
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use')
    
    return parser.parse_args()


def setup_environment(args):
    """Setup environment based on arguments."""
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_dict = vars(args)
    config_dict['device'] = str(device)
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f'Device: {device}')
    print(f'Config written → {output_dir / "config.json"}')
    
    return device, output_dir


def get_config_dict(args):
    """Convert args to config dictionary."""
    return {
        "features_prefix": args.features_prefix,
        "label_col": args.label_col,
        "chan_col": args.chan_col,
        "subset_col": args.subset_col,
        "time_col": args.time_col,
        "window": args.window,
        "ae_window": args.ae_window,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "lr": args.lr,
        "hidden": args.hidden,
        "n_layers": args.n_layers,
        "dropout": args.dropout,
        "z_dim": args.z_dim,
        "error_mode": args.error_mode,
        "area_l": args.area_l,
        "ewma_alpha": args.ewma_alpha,
        "gmm_components_max": args.gmm_components_max,
        "target_fpr": args.target_fpr,
        "min_p": args.min_p,
        "min_event_len": args.min_event_len,
        "dilate_radius": args.dilate_radius,
        "num_workers": args.num_workers,
        "seed": args.seed,
    }
